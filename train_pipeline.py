import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torch.autograd import Variable

import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback

import torch.nn.functional as F

from datasets import  DVSGesture, EventReID_Dataset, SEEDataset
from networks import PipelineNet, ClassNet
from losses import TripletLoss, AntiTripletLoss, entropy_loss
from utils import fliplr, id_evaluate, plot_confusion_matrix, plot_event_frame

class PipelineModule(pl.LightningModule):

    def __init__(self, args, train_dataset, val_dataset):
        super(PipelineModule, self).__init__()
        self.args = args
        self.dataset = self.args.dataset
        self.automatic_optimization = False
        
        if self.dataset in ['dvsg', 'see']:
            self.input_c = 10

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.n_classes = len(self.train_dataset.label_map_target)
        self.n_target_classes = len(self.train_dataset.label_map_target)
        self.n_id_classes = len(self.train_dataset.label_map_id)
        self.model = PipelineNet(self.args, self.n_target_classes, self.n_id_classes, self.input_c)
        
        self.gamma = 2
        self.full_eval_metrics = []

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle = True, num_workers= self.args.num_workers)
        return train_loader

    def val_dataloader(self):
        target_val_dataset, query_dataset, gallery_dataset = self.val_dataset
        target_val_loader = DataLoader(target_val_dataset, batch_size=self.args.batch_size, num_workers= self.args.num_workers)
        query_loader = DataLoader(query_dataset, batch_size=self.args.batch_size, num_workers= self.args.num_workers)
        gallery_loader = DataLoader(gallery_dataset, batch_size=self.args.batch_size, num_workers= self.args.num_workers)

        return [target_val_loader, query_loader, gallery_loader]

    def training_step(self, batch, batch_idx):
        x, target, user_id, file_id = batch
        x_anno, id_pred, id_embed_feature, id_pool5_feature, target_pred, id_pred2, id_embed_feature2, id_pool5_feature2, target_pred2= self.model(x)
        
        target_loss = nn.CrossEntropyLoss()(target_pred, target)
        target_loss2 = nn.CrossEntropyLoss()(target_pred2, target)
        preds = torch.argmax(target_pred, dim=1)
        target_accuracy = (preds == target).float().mean()
        self.log('train_target/accuracy', target_accuracy, prog_bar=True, on_epoch=True)
        self.log('train_target/loss', target_loss, on_epoch=True)

        
        ce_loss2 = nn.CrossEntropyLoss()(id_pred2, user_id)
        triplet2 = TripletLoss(margin = 4)
        triplet_loss2 = triplet2(id_embed_feature2, user_id)
        id_loss = ce_loss2 + triplet_loss2
        self.log('train_id/loss', id_loss, on_epoch=True)
                
        triplet = TripletLoss(margin = 4)
        triplet_loss = triplet(id_embed_feature, user_id)
            
        mean_target =  torch.ones_like(id_pred) * (1/self.n_id_classes)
        mean_loss = nn.CrossEntropyLoss()(id_pred, mean_target)

        preds = torch.argmax(id_pred, dim=1)
        id_accuracy = (preds == user_id).float().mean()
        self.log('train_id/id_accuracy', id_accuracy, prog_bar=True, on_epoch=True)
            
        optimizer1, optimizer2, optimizer3 = self.optimizers()
        
        scheduler1, scheduler2, scheduler3 = self.lr_schedulers()
        
        self.toggle_optimizer(optimizer1)
        loss1 = target_loss - triplet_loss
        self.log('train_full/loss_target', loss1, on_epoch=True)
            
        optimizer1.zero_grad()
        self.manual_backward(loss1)
        optimizer1.step()
        self.untoggle_optimizer(optimizer1)

        # target network
        self.toggle_optimizer(optimizer3)
        optimizer3.zero_grad()
        self.manual_backward(target_loss2)
        optimizer3.step()
        self.untoggle_optimizer(optimizer3)

        # id network
        self.toggle_optimizer(optimizer2)
        loss2 = id_loss 
        self.log('train_full/loss_id', id_loss, on_epoch=True)
        optimizer2.zero_grad()
        self.manual_backward(loss2)
        optimizer2.step()
        self.untoggle_optimizer(optimizer2)
        
        if self.trainer.is_last_batch:
            scheduler1.step()
            scheduler2.step()
            scheduler3.step()


    def validation_step(self, batch, batch_idx, dataloader_idx= 0):
        x, target, user_id, file_id = batch
        

        if dataloader_idx == 0:

            x_anno, id_pred, id_embed_feature, id_pool5_feature, target_pred, id_pred2, id_embed_feature2, id_pool5_feature2, target_pred2 = self.model(x)
            
            ce_loss = torch.nn.functional.cross_entropy(target_pred, target, reduction='none')
            pt = torch.exp(-ce_loss)
            gamma = 2.0
            alpha = 0.25
            target_val_loss = (alpha * (1-pt)**gamma * ce_loss).mean()

            preds = torch.argmax(target_pred, dim=1)
            target_accuracy = (preds == target).float().mean()

            for true_label, predicted_label in zip(target.view(-1), preds.view(-1)):
                self.confusion_matrix[true_label, predicted_label] += 1

            self.log('val_target/loss', target_val_loss)
            self.log('val_target/accuracy', target_accuracy, prog_bar=True)
            
            self.full_target_acc.append(target_accuracy)

            if batch_idx in [2,3]: #12,13
                for i in range(5):
                    ev_tmp = x[i].cpu()
                    ev_tmp2 = x_anno[i].cpu()
                    tensorboard = self.logger.experiment
                    plot_event_frame(ev_tmp, ev_tmp2, i, tensorboard, self.current_epoch, name=f'{self.args.ename}_{batch_idx}')
                    
        if dataloader_idx > 0:
            n, c, h, w = x.size()
            for i in range(2):
                if(i==1):
                    x = fliplr(x)
                input_img = Variable(x.to(self.device))
                x_anno, id_pred, id_embed_feature, id_pool5_feature, target_pred, id_pred2, id_embed_feature2, id_pool5_feature2, target_pred2 = self.model(input_img)

                if(i==0):
                    ff_pool5 = torch.FloatTensor(n,id_pool5_feature.size(1)).zero_()
                    ff_embed = torch.FloatTensor(n,id_embed_feature.size(1)).zero_()
                f_pool5 = id_pool5_feature.data.cpu()
                ff_pool5 = ff_pool5 + f_pool5
                f_embed = id_embed_feature.data.cpu()
                ff_embed = ff_embed + f_embed
                    
            fnorm_pool5 = torch.norm(ff_pool5, p=2, dim=1, keepdim=True)
            fnorm_embed = torch.norm(ff_embed, p=2, dim=1, keepdim=True)
            ff_pool5 = ff_pool5.div(fnorm_pool5.expand_as(ff_pool5))
            ff_embed = ff_embed.div(fnorm_embed.expand_as(ff_embed))
            if dataloader_idx == 1:
                self.query_pool5_features = torch.cat((self.query_pool5_features,ff_pool5), 0)
                self.query_embed_features = torch.cat((self.query_embed_features,ff_embed), 0)
                self.query_label.extend(user_id.tolist())
                self.query_target.extend(target.tolist())
                self.query_file_id.extend(list(file_id))
            elif dataloader_idx == 2:
                self.gallery_pool5_features = torch.cat((self.gallery_pool5_features,ff_pool5), 0)
                self.gallery_embed_features = torch.cat((self.gallery_embed_features,ff_embed), 0)
                self.gallery_label.extend(user_id.tolist())
                self.gallery_target.extend(target.tolist())
                self.gallery_file_id.extend(list(file_id))


      
            
    def on_validation_epoch_start(self):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes), dtype=torch.int64)
        
        self.query_pool5_features = torch.FloatTensor()
        self.query_embed_features = torch.FloatTensor()
        self.gallery_pool5_features = torch.FloatTensor()
        self.gallery_embed_features = torch.FloatTensor()
        self.query_label = []
        self.gallery_label = []
        self.query_file_id = []
        self.gallery_file_id = []
        self.query_target = []
        self.gallery_target = []
        
        self.full_target_acc = []
        self.full_id_mAP = 0

    def on_validation_epoch_end(self):
        #target
        unique_labels_target = self.train_dataset.unique_labels_target
        cm = self.confusion_matrix.numpy()
        tensorboard = self.logger.experiment
        plot_confusion_matrix(cm, unique_labels_target, tensorboard, self.current_epoch)
        
        #id
        CMC = torch.IntTensor(len(self.gallery_label)).zero_()
        ap = 0.0
        for i in range(len(self.query_label)):
            ap_tmp, CMC_tmp = id_evaluate(self.query_embed_features[i], self.query_label[i], self.query_target[i], self.query_file_id[i],
                                    self.gallery_embed_features, self.gallery_label, self.gallery_target, self.gallery_file_id)
            if CMC_tmp[0]==-1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp

        CMC = CMC.float()
        CMC = CMC/len(self.query_label) #average CMC
        #print(CMC, flush = True)
        print('Pool5-Feature top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(self.query_label)))
        self.log('val_id/accuracy', CMC[0])
        self.log('val_id/accuracy5', CMC[4])
        self.log('val_id/accuracy10', CMC[9])
        self.log('val_id/mAP', ap/len(self.query_label))#
       
        self.full_id_mAP = ap/len(self.query_label)

        
    
    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(self.model.anonnet.parameters(), lr= self.args.lr0)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, self.trainer.max_epochs)

        optimizer2 = torch.optim.Adam(self.model.idnet.parameters(), lr=self.args.lr0_helper)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=100, gamma=0.5)
        
        optimizer3 = torch.optim.Adam(self.model.targetnet.parameters(), lr= self.args.lr0_helper)
        scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=100, gamma=0.5)
        
        return [optimizer1, optimizer2, optimizer3],  [scheduler1, scheduler2, scheduler3]

class SaveBestModelWeights(Callback):
    def __init__(self, dirpath, monitor='val_accuracy', monitor_op = 'max', args = None):
        super(SaveBestModelWeights, self).__init__()
        self.monitor = monitor
        self.args = args
        self.monitor_op = monitor_op
        self.best_metric_value = float('inf') if self.monitor_op == 'min' else float('-inf')
        self.dirpath = dirpath
        os.makedirs(self.dirpath, exist_ok=True)
    

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch > 0:
            current_metric_value = trainer.callback_metrics.get(self.monitor)
            if current_metric_value is not None:
                if (self.monitor_op == "min" and current_metric_value < self.best_metric_value) or \
                (self.monitor_op == "max" and current_metric_value > self.best_metric_value):
                    self.best_metric_value = current_metric_value
                    
                    os.makedirs(self.dirpath+'/target', exist_ok=True)
                    torch.save(pl_module.model.targetnet.state_dict(), f'{self.dirpath}/target/e{trainer.current_epoch}_best_model_weights.pth')
                    os.makedirs(self.dirpath+'/id', exist_ok=True)
                    torch.save(pl_module.model.idnet.state_dict(), f'{self.dirpath}/id/e{trainer.current_epoch}_best_model_weights.pth')
                    os.makedirs(self.dirpath+'/anon', exist_ok=True)
                    torch.save(pl_module.model.anonnet.state_dict(), f'{self.dirpath}/anon/e{trainer.current_epoch}_best_model_weights.pth')
                if trainer.current_epoch == trainer.max_epochs -1:
                    os.makedirs(self.dirpath+'/target', exist_ok=True)
                    torch.save(pl_module.model.targetnet.state_dict(), f'{self.dirpath}/target/e{trainer.current_epoch}_last_epoch_weights.pth')
                    os.makedirs(self.dirpath+'/id', exist_ok=True)
                    torch.save(pl_module.model.idnet.state_dict(), f'{self.dirpath}/id/e{trainer.current_epoch}_last_epoch_weights.pth')
                    os.makedirs(self.dirpath+'/anon', exist_ok=True)
                    torch.save(pl_module.model.anonnet.state_dict(), f'{self.dirpath}/anon/e{trainer.current_epoch}_last_epoch_weights.pth')
                    
# ArgumentParser setup
parser = argparse.ArgumentParser(description='Event Ethics')
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs for training')
parser.add_argument('--batch_size', default=16, type=int, help='batch size for training')
parser.add_argument('--num_workers', default=6, type=int, help='number of workers')
parser.add_argument('--dataset', default='rgb', type=str, choices = ['dvsg', 'see'], help='dataset to be used')
parser.add_argument('--dataset_path', default='./data/DVSGesture', type=str, help='path to dataset')
parser.add_argument('--network', default='resnet50', type=str, choices = ['resnet50'])
parser.add_argument('--lr0', default=1e-4, type=float)
parser.add_argument('--lr0_helper', default=1e-4, type=float)
parser.add_argument('--ename', default='experiment', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--store_weight', action='store_true')
parser.add_argument('--target_weights', default=None, type=str)
parser.add_argument('--id_weights', default=None, type=str)
parser.add_argument('--denoise_weights', default=None, type=str)
parser.add_argument('--val_only', action='store_true')
parser.add_argument('--noise_std', default=1.0, type=float)
args = parser.parse_args()

seed = 42
pl.seed_everything(seed, workers = True)
    
if args.dataset == 'dvsg':
    train_transform = transforms.Compose([transforms.RandomCrop(128, padding=128 // 12), transforms.RandomRotation(degrees=15)])
    train_dataset = DVSGesture(mode = 'train', main_dir = args.dataset_path, transform = train_transform)
    
    target_val_dataset = DVSGesture(mode = 'val', main_dir = args.dataset_path)
    query_dataset = DVSGesture(mode ='query', main_dir = args.dataset_path)
    gallery_dataset = DVSGesture(mode ='gallery', main_dir = args.dataset_path)
    val_dataset = [target_val_dataset, query_dataset, gallery_dataset]
elif args.dataset == 'see':
    train_transform = transforms.Compose([transforms.RandomCrop(180, padding=180 // 12), 
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomRotation(degrees=15)])
    train_dataset = SEEDataset(mode = 'train', main_dir = args.dataset_path, transform = train_transform)
    target_val_dataset = SEEDataset(mode = 'val', main_dir = args.dataset_path)
    query_dataset = SEEDataset(mode ='query', main_dir = args.dataset_path)
    gallery_dataset = SEEDataset(mode ='gallery', main_dir = args.dataset_path)
    val_dataset = [target_val_dataset, query_dataset, gallery_dataset]
        

model = PipelineModule(args=args, train_dataset=train_dataset, val_dataset = val_dataset)

#we will still take the last checkpoint for posttraining
monitor_metric = 'val_target/accuracy/dataloader_idx_0' 
monitor_mode = 'max'

checkpoint_callback = ModelCheckpoint(
    dirpath=f'./ckpt/{args.ename}',
    filename='{epoch}-{val_loss:.2f}',
    save_top_k=5,
    verbose=True,
    monitor=monitor_metric,
    mode=monitor_mode
)

lr_monitor = LearningRateMonitor(logging_interval='step')

callbacks = [checkpoint_callback, lr_monitor]

if args.store_weight:
    callbacks.append(SaveBestModelWeights(monitor=monitor_metric, dirpath=f'./weights/{args.ename}',monitor_op = monitor_mode, args = args))
    
trainer = pl.Trainer(
    max_epochs=args.epochs,
    devices=1,
    accelerator="gpu",
    deterministic= True, 
    logger=TensorBoardLogger('logs/', name= args.ename, version = 0),
    callbacks= callbacks,
    check_val_every_n_epoch= 1, 
    precision=32
)

if args.val_only:
    trainer.validate(model)
else:
    trainer.fit(model, ckpt_path=args.resume)
