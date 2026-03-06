import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
import numpy as np
from pytorch_wavelets import DWTForward

class OHEMBCEWithLogits(nn.Module):
    def __init__(self, keep_ratio=0.7):
        super(OHEMBCEWithLogits, self).__init__()
        self.keep_ratio = keep_ratio
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets, weights=None):
        loss = self.bce(inputs, targets)
        if weights is not None:
            loss = loss * weights
    
        loss_flat = loss.view(-1)
        num_keep = int(self.keep_ratio * loss_flat.numel())
        val, _ = torch.topk(loss_flat, num_keep)
        
        return val.mean()


def compute_weights(gt, w0=15.0, sigma=3.0):
    gt_numpy = gt.cpu().numpy().astype(np.uint8)
    weights = np.zeros_like(gt_numpy, dtype=np.float32)
    
    for i in range(gt.shape[0]):
      
        dist = distance(gt_numpy[i, 0] == 0)
        
        
        w_c = np.ones_like(gt_numpy[i, 0], dtype=np.float32)
        class_counts = np.bincount(gt_numpy[i, 0].flatten())
        
        if len(class_counts) >= 2:
            
            total = class_counts.sum()
            w_c[gt_numpy[i, 0] == 0] = total / (class_counts[0] + 1e-6)
            w_c[gt_numpy[i, 0] == 1] = (total / (class_counts[1] + 1e-6)) * 2.0 # 额外给血管2倍关注
        
    
        w_dist = w0 * np.exp(-((dist)**2 / (2 * sigma**2)))
        weights[i, 0] = w_c + w_dist
   
    return torch.from_numpy(weights).float().to(gt.device)


import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSegmentationLoss(nn.Module):
    def __init__(self, 
                 w_bce=0.3, 
                 w_dice=1.5, 
                 w_tversky=1.5, 
                 w_focal=0.8, 
                 tversky_alpha=0.7, 
                 tversky_beta=0.3,
                 focal_alpha=0.25, 
                 focal_gamma=1.5,
                 **kwargs): 
        super(EnhancedSegmentationLoss, self).__init__()
        
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_tversky = w_tversky
        self.w_focal = w_focal
       
        
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def tversky_loss(self, score, target):
        smooth = 1e-5
    
        tp = torch.sum(score * target)
        fp = torch.sum(score * (1 - target))
        fn = torch.sum((1 - score) * target)
        
        tversky = (tp + smooth) / (tp + self.tversky_alpha * fn + self.tversky_beta * fp + smooth)
        return 1 - tversky

    def focal_loss(self, score, target):
        smooth = 1e-5
        pos_loss = -self.focal_alpha * torch.pow(1 - score, self.focal_gamma) * target * torch.log(score + smooth)
        neg_loss = -(1 - self.focal_alpha) * torch.pow(score, self.focal_gamma) * (1 - target) * torch.log(1 - score + smooth)
        return torch.mean(pos_loss + neg_loss)

    def forward(self, inputs, target):
        
        bce = F.binary_cross_entropy_with_logits(inputs, target)
        
        probs = torch.sigmoid(inputs)
    
        dice = self.dice_loss(probs, target)
        tversky = self.tversky_loss(probs, target)
        focal = self.focal_loss(probs, target)

        total_loss = (self.w_bce * bce + 
                      self.w_dice * dice + 
                      self.w_tversky * tversky + 
                      self.w_focal * focal)
        
        loss_details = {
            'total_loss': total_loss.item(),
            'bce_loss': bce.item(),
            'dice_loss': dice.item(),
            'tversky_loss': tversky.item(),
            'focal_loss': focal.item()
        }
        
        return total_loss, loss_details