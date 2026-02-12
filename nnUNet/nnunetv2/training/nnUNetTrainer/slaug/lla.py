# lla.py
import torch
from bezier import intensity_remap 

@torch.no_grad()
def to_label_map(y: torch.Tensor) -> torch.Tensor:
    """
    Accepts:
      - (B, 1, ...) or (B, ...) integer tensor
      - (B, C, ...) one-hot (optional support)
    Returns:
      - (B, ...) int64 label map
    """
    if y.ndim >= 2 and y.shape[1] == 1:
        # (B,1,...) -> (B,...)
        y = y[:, 0]
    if y.ndim >= 2 and y.dtype.is_floating_point and y.shape[1] > 1:
        # one-hot -> argmax
        y = torch.argmax(y, dim=1)
    return y.long()


@torch.no_grad()
def lla_augment(
    x: torch.Tensor,
    y: torch.Tensor,
    classes: list[int],
    num_knots: int = 8,
    strength: float = 0.25,
    apply_to_background: bool = False,
):
    """
    Local Location-scale Augmentation (LLA), MVP version.

    Args:
      x: (B, C, ...) float tensor, assumed normalized to [0,1]
      y: segmentation labels (B,1,...) or (B,...) int
      classes: list of class ids to augment (e.g., [1,2,3] excluding background 0)
      apply_to_background: if True, also remap background (class 0)

    Returns:
      x_lla: (B, C, ...) float tensor
    """
    assert x.ndim >= 3, "Expect (B,C,spatial...)"
    ylab = to_label_map(y)  # (B, spatial...)

    B, C = x.shape[:2]
    spatial_shape = x.shape[2:]

    # output init: start as original (so uncovered voxels remain original)
    out = x.clone()

    # optional: background remap
    if apply_to_background:
        mask0 = (ylab == 0).unsqueeze(1)  # (B,1,...)
        x_bg = intensity_remap(x, num_knots=num_knots, strength=strength)
        out = torch.where(mask0, x_bg, out)

    # per-class remap inside each mask
    for c in classes:
        mask = (ylab == c).unsqueeze(1)  # (B,1,...), bool
        if not mask.any():
            continue

        # remap whole x then select only mask region (simple MVP)
        # (optimization later: only remap masked voxels)
        x_c = intensity_remap(x, num_knots=num_knots, strength=strength)

        # write back only where mask==True
        out = torch.where(mask, x_c, out)

    return out


def test_lla_augment():
    x = torch.rand(4, 1, 128, 128, 128).cuda()
    y = torch.randint(0, 5, (4, 128, 128, 128)).cuda()
    classes = [1, 2, 3, 4]
    x_lla = lla_augment(x, y, classes, num_knots=8, strength=0.25, apply_to_background=True)
    print(x_lla.shape)
    assert torch.isfinite(x_lla).all()
    assert x_lla.min() >= 0.0 - 1e-5
    assert x_lla.max() <= 1.0 + 1e-5

if __name__ == "__main__":
    import nibabel as nib
    import numpy as np
    import torch
    x = nib.load("/data2/pyq6817/nnUNetv2/nnUNet_raw/Dataset404_AbdCT/imagesTr/Case_00002_0000.nii.gz").get_fdata()
    y = nib.load("/data2/pyq6817/nnUNetv2/nnUNet_raw/Dataset404_AbdCT/labelsTr/Case_00002.nii.gz").get_fdata()
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    y = y.unsqueeze(0)
    print(x.shape, y.shape)
    x_lla = lla_augment(x, y, classes=[1, 2, 3, 4, 5], num_knots=8, strength=0.25, apply_to_background=True)
    print(x_lla.shape)
    x_lla = x_lla.squeeze(0)
    x_lla = x_lla.squeeze(0)
    nib.save(nib.Nifti1Image(x_lla.cpu().numpy(), np.eye(4)), "x_lla.nii.gz")