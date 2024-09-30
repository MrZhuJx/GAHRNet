import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from torch import einsum
# from timm.models.layers.helpers import to_2tuple
from timm.models.layers import DropPath, trunc_normal_
import math





class KpLoss(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        
        device = logits.device
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = targets.to(device)
        
        
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        # print('loss=',loss.size())
        loss = torch.sum(loss * 1.0) / bs  
        return loss
    
class KpLoss3(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        
        device = logits.device
        a = torch.tensor([1.50,1.19,1.15 ,1.11 ,1.37 ,1.50 ,1.01 ,1.11 ,1.12 ,1.00 ,1.17 ,1.34 ,1.36 ,1.07,
                          1.46 ,1.02 ,1.09 ,1.50 ,1.42 ,1.29 ,1.27 ,1.38 ,1.01 ,1.77 ,1.51 ,1.32 ,1.35 ,1.12 ,1.13]).to(device)
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = targets.to(device)
        
        
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        # print('loss=',loss.size())
        loss = torch.sum(loss * a) / bs  
        return loss


class KpLoss1(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets, logits2, targets2):
        
        device = logits.device
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = targets.to(device)
        heatmaps2 = targets2.to(device)
        
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        # print('loss=',loss.size())
        loss = torch.sum(loss * 1.0) / bs  

        loss2 = self.criterion(logits2, heatmaps2).mean(dim=[2, 3])
        loss2 = torch.sum(loss2 * 1.0) / bs
        return loss + loss2
    

    
if __name__ == '__main__':

    model = KpLoss()

    x = torch.rand(8,19,168,168)
    x1 = torch.rand(8,19,168,168)
    x2 = torch.rand(8,128,42,42).cuda('cuda:1')
    x3 = torch.rand(8,3,672,672).cuda('cuda:1')
    
    y1 = model(x,x1)

    
    print(y1.size())
    
    
