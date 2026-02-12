import torch.nn as nn
import torch
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_hat,y=None):
        assert x.shape == x_hat.shape, "x and x_hat must have the same shape"
        assert y is None or y.shape == x.shape, "y must have the same shape as x"
        # if y is None, simply return the l1 loss between x and x_hat
        if y is None:
            diff = torch.abs(x_hat - x)      # (B,C,D,H,W)
            loss = (diff).sum() / (x.shape[2] * x.shape[3] * x.shape[4] + 1e-6)
        # if y is not None, masked reconstruction loss
        else:
            fg_mask = (y > 0).float()
            diff = torch.abs(x_hat - x)      # (B,C,D,H,W)
            loss = (diff * fg_mask).sum() / (fg_mask.sum() + 1e-6)
        return loss

if __name__ == "__main__":
    x = torch.randn(1, 1, 128, 128, 128)
    x_hat = torch.randn(1, 1, 128, 128, 128)
    y = torch.randn(1, 1, 128, 128, 128)
    loss = ReconstructionLoss()
    loss_value = loss(x, x_hat,y)
    print(loss_value)