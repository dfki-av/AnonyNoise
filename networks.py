import torch 
from torch import nn
from torch.nn import init
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib.colors import hsv_to_rgb

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class PostClassNet(nn.Module):
    def __init__(self, network = 'resnet50', n_classes= 7, input_c = 3, args = None):
        super(PostClassNet, self).__init__()
        self.args = args
        self.input_c = input_c
        self.classnet = ClassNet(network, n_classes, input_c)
        if self.args.class_weights is not None:
            self.classnet.load_state_dict(torch.load(self.args.class_weights))

        if self.args.denoisenet:
            self.denoisenet = DenoiseNet(input_c)
        
        self.anonnet = SimpleAnonNet(input_c, args)
        if self.args.anon_weights is not None:
            self.anonnet.load_state_dict(torch.load(self.args.anon_weights))
            
        for param in self.anonnet.parameters():
            param.requires_grad = False
        
    
    def forward(self, x):
        
        x = self.anonnet(x)
        if self.args.denoisenet:
            x_denoise = self.denoisenet(x)
            pred, feature1, feature2 = self.classnet(x_denoise)
            return pred, feature1, feature2, x_denoise
        else:
            pred, feature1, feature2 = self.classnet(x) 
            return pred, feature1, feature2, None
       
        
    
class ClassNet(nn.Module):
    def __init__(self, network = 'resnet50', n_classes= 7, input_c = 3, class_weights = None):
        super(ClassNet, self).__init__()

        if network == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights)
            out_dim = 2048
        if network == 'resnet34':
            backbone = models.resnet34(weights=models.ResNet34_Weights)
            out_dim = 512
        if network == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights)
            out_dim = 512
    
        if 'resnet' in network:
            if input_c > 3:
                weight = backbone.conv1.weight.clone()
                backbone.conv1 = nn.Conv2d(input_c, 64, kernel_size=7, stride=2, padding=3, bias=False)
                with torch.no_grad():
                    backbone.conv1.weight[:, :3] = weight
                    for i in range(input_c):
                        backbone.conv1.weight[:, i] = backbone.conv1.weight[:, i%3]

            modules = list(backbone.children())[:-1]
    
            self.backbone = nn.Sequential(*modules)

        self.linear_block = nn.Sequential(
            nn.Linear(out_dim, 512),
            nn.BatchNorm1d(512),  
            nn.LeakyReLU(0.5),
            nn.Dropout(p=0.5) 
        )
        if class_weights is None:
            self.linear_block.apply(weights_init_kaiming)
            
        self.classifier = nn.Sequential(
            nn.Linear(512, n_classes)
        )
        if class_weights is None:
            self.classifier.apply(weights_init_classifier)
        
        if class_weights is not None:
            self.load_state_dict(torch.load(class_weights))
                   
                
    def forward(self, x):
        x = self.backbone(x)
        x = torch.squeeze(x)
        feature = self.linear_block(x)
        pred = self.classifier(feature)
        return pred, feature, x

    
class PipelineNet(torch.nn.Module):
    def __init__(self, args = None, n_target_classes = 0, n_id_classes = 0, input_c = 3):
        super(PipelineNet, self).__init__()
        self.args = args
        self.num_mus = 6
        self.n_target_classes = n_target_classes
        self.n_id_classes = n_id_classes
        self.input_c = input_c
        self.targetnet = ClassNet(self.args.network, self.n_target_classes, self.input_c)
        if self.args.target_weights is not None:
            self.targetnet.load_state_dict(torch.load(self.args.target_weights))
            
        self.idnet = ClassNet(self.args.network, self.n_id_classes , self.input_c)
        if self.args.id_weights is not None:
            self.idnet.load_state_dict(torch.load(self.args.id_weights))
        
        self.anonnet = SimpleAnonNet(self.input_c,args)
        
    def forward(self, x):
        x_anno = self.anonnet(x) 
        
        id_pred, id_embed_feature, id_pool5_feature = self.idnet(x_anno)
        id_pred2, id_embed_feature2, id_pool5_feature2 = self.idnet(x_anno.detach())
        
        target_pred, _, _ = self.targetnet(x_anno)
        target_pred2, _, _ = self.targetnet(x_anno.detach())
        
        return x_anno, id_pred, id_embed_feature, id_pool5_feature, target_pred, id_pred2, id_embed_feature2, id_pool5_feature2, target_pred2
    
class DenoiseNet(nn.Module):
    def __init__(self, channels=2, num_layers=15):
        super(DenoiseNet, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(128))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(in_channels=128, out_channels=channels, kernel_size=3, padding=1))
        
        self.dncnn = nn.Sequential(*layers)
                
    def forward(self, x):
        return self.dncnn(x)

    
class SimpleAnonNet(nn.Module):
    def __init__(self, in_channels, args):
        super(SimpleAnonNet, self).__init__()
        self.in_channels = in_channels
        self.args = args
        input_channels = self.in_channels *2
        
        self.convs1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
    
        self.convs2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
 
        out_channels = self.in_channels
            
        self.convs3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        noise = torch.randn_like(x)*self.args.noise_std
            
        x_noise = torch.cat((x, noise), dim = 1)
        x_ = self.convs1(x_noise)
        x_ = self.convs2(x_)
        x_ = self.convs3(x_)
        res = x_ * noise
        res = x + res
        return  res
