# saliency.py
import torch
import torch.nn.functional as F
import torch.nn as nn

def _reduce_channel(saliency: torch.Tensor, mode: str = "mean"):
    """
    saliency: (B, C, ...)
    returns:  (B, 1, ...)
    """
    if saliency.ndim < 3:
        raise ValueError("Expect saliency with shape (B,C,...)")

    if mode == "mean":
        return saliency.mean(dim=1, keepdim=True)
    elif mode == "max":
        return saliency.max(dim=1, keepdim=True).values
    else:
        raise ValueError(f"Unknown channel reduction: {mode}")


def compute_saliency(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_type: str = "ce",
    channel_reduce: str = "mean",
):
    """
    Compute input-gradient saliency map.

    Args:
      model: frozen segmentation model (eval mode!)
      x: (B, C, ...) input tensor (GLA output)
      y: segmentation labels (B,1,...) or (B,...)
      loss_type: 'ce' or 'logit'
      channel_reduce: 'mean' or 'max'

    Returns:
      saliency: (B, 1, ...) float tensor
    """
    model.eval()

    # make sure model params are frozen
    for p in model.parameters():
        p.requires_grad_(False)

    x = x.detach().clone()
    x.requires_grad_(True)

    # forward
    logits = model(x)  # nnUNet output: (B, num_classes, ...)

    if loss_type == "ce":
        # Cross-entropy needs (B, ...) labels
        if y.ndim >= 2 and y.shape[1] == 1:
            y_ce = y[:, 0]
        else:
            y_ce = y
        loss = F.cross_entropy(logits, y_ce.long())
    elif loss_type == "logit":
        # sum of foreground logits (task-agnostic & cheap)
        loss = logits.sum()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # backward: gradient w.r.t input
    loss.backward()

    # saliency = |dL/dx|
    sal = x.grad.abs()  # (B, C, ...)

    sal = _reduce_channel(sal, mode=channel_reduce)

    return sal.detach()


def smooth_saliency(saliency: torch.Tensor, grid_size: int):
    """
    Grid-based saliency smoothing using avg pooling.

    Args:
      saliency: (B,1,H,W) or (B,1,D,H,W)
      grid_size: g (int)

    Returns:
      smoothed saliency, same shape
    """
    if grid_size <= 1:
        return saliency

    # Use grid_size//2 padding to maintain size (works for both odd and even)
    pad = grid_size // 2
    
    if saliency.ndim == 4:
        # 2D
        smoothed = F.avg_pool2d(
            saliency,
            kernel_size=grid_size,
            stride=1,
            padding=pad,
        )
        # Crop to original size (handles both odd and even grid_size)
        _, _, h, w = saliency.shape
        return smoothed[:, :, :h, :w].contiguous()
    elif saliency.ndim == 5:
        # 3D
        smoothed = F.avg_pool3d(
            saliency,
            kernel_size=grid_size,
            stride=1,
            padding=pad,
        )
        # Crop to original size (handles both odd and even grid_size)
        _, _, d, h, w = saliency.shape
        return smoothed[:, :, :d, :h, :w].contiguous()
    else:
        raise ValueError(f"Unsupported saliency shape: {saliency.shape}")


def test_compute_saliency():
    model = nn.Sequential(
        nn.Conv3d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv3d(16, 16, kernel_size=3, padding=1),
        nn.ReLU(),)
    model.eval()
    x = torch.rand(4, 1, 128, 128, 128)
    y = torch.randint(0, 5, (4, 128, 128, 128))
    saliency = compute_saliency(model, x, y)
    print(saliency.shape)
    smoothed_saliency = smooth_saliency(saliency, grid_size=4)
    print(smoothed_saliency.shape)


if __name__ == "__main__":
    test_compute_saliency()