import torch 
from torch import nn
from torch.autograd import Variable

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta = 1, alpha= -2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    masked_distances_ap = dist_mat.masked_fill(~is_pos, float('-inf'))
    dist_ap = torch.max(masked_distances_ap, dim=1).values
    
    masked_distances_an = dist_mat.masked_fill(~is_neg, float('inf'))
    dist_an = torch.min(masked_distances_an, dim=1).values

    return dist_ap, dist_an


class TripletLoss(object):

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.data.new().resize_as_(dist_an.data).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, Variable(y))
        else:
            loss = self.ranking_loss(dist_an - dist_ap, Variable(y))
        return loss #, dist_ap, dist_an
    
class AntiTripletLoss(object):

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        
        ###### version 0
        # target_mat = torch.ones_like(dist_mat)*1e-12
        # dist_mean = torch.mean(dist_mat)
        # n = dist_mat.size(0)
        # target_mat[torch.arange(n)[:, None] != torch.arange(n)] = dist_mean
        
        ###### version 0
        target_mat = 0.8 * dist_mat
        loss = nn.functional.mse_loss(dist_mat, target_mat)
        return -loss #, dist_ap, dist_an
    
def entropy_loss(input):
    input = input + 1e-16  # Add a small constant for numerical stability
    probabilities = nn.functional.softmax(input, dim=1)
    log_probabilities = torch.log(probabilities)
    
    # Calculate the entropy loss
    H = -torch.mean(torch.sum(probabilities * log_probabilities, dim=1))
    
    return H
