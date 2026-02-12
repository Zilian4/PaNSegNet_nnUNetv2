import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss

class BoundaryGuidedLoss(nn.Module):
    def __init__(self,seg_loss:nn.Module, boundary_loss:nn.Module, lambda_bg:float=0.2):
        super().__init__()
        self.seg_loss = seg_loss
        self.boundary_loss = boundary_loss
        self.lambda_bg = lambda_bg
    
    def forward(self, 
                seg_output: torch.Tensor, 
                seg_target: torch.Tensor,
                boundary_output: torch.Tensor,
                boundary_target: torch.Tensor) -> dict:
        seg_loss = self.seg_loss(seg_output, seg_target)
        boundary_loss = self.boundary_loss(boundary_output, boundary_target)
        total_loss = seg_loss*(1-self.lambda_bg) + boundary_loss*self.lambda_bg
        return {
            'total_loss': total_loss,
            'segmentation_loss': seg_loss,
            'boundary_loss': boundary_loss,
        }

    def set_lambda_bg(self, lambda_bg: float):
        self.lambda_bg = lambda_bg  
        


if __name__ == "__main__":
    # Test the domain adaptation loss
    seg_output = torch.randn(3, 4, 128, 128, 128)
    seg_target = torch.randint(0, 4, (3, 1, 128, 128, 128))
    boundary_output = torch.randn(3, 1, 128, 128, 128)
    boundary_target = torch.randint(0, 1, (3, 1, 128, 128, 128))
    seg_loss = DC_and_CE_loss(
        soft_dice_kwargs={'batch_dice': True,'ddp': False},
        ce_kwargs={},
        weight_ce=1,
        weight_dice=1
    )
    # Boundaary loss is dice for boundary
    boundary_loss = DC_and_BCE_loss(
        soft_dice_kwargs={'batch_dice': True,'ddp': False},
        bce_kwargs={},
        weight_ce=1,
        weight_dice=1
    )
    loss = BoundaryGuidedLoss(seg_loss=seg_loss, boundary_loss=boundary_loss, lambda_bg=0.2)
    loss_dict = loss(seg_output, seg_target, boundary_output, boundary_target)
    print(loss_dict)