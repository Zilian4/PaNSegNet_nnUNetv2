# bezier.py
import torch

@torch.no_grad()
def sample_monotone_curve(num_knots: int = 8,
                          strength: float = 0.25,
                          device=None,
                          dtype=torch.float32):
    """
    Returns (xp, fp): a monotone increasing mapping defined by knots.
    xp: fixed x locations in [0,1]
    fp: sampled y values in [0,1], sorted to enforce monotonicity
    strength controls how far fp can deviate from identity.
    """
    assert num_knots >= 2
    xp = torch.linspace(0.0, 1.0, steps=num_knots, device=device, dtype=dtype)

    # start from identity then perturb
    base = xp.clone()
    noise = (torch.rand_like(base) - 0.5) * 2.0  # [-1,1]
    fp = base + strength * noise

    # enforce endpoints and range
    fp[0] = 0.0
    fp[-1] = 1.0
    fp = fp.clamp(0.0, 1.0)

    # enforce monotonicity by sorting (simple & stable)
    fp, _ = torch.sort(fp)

    # ensure endpoints exactly
    fp[0] = 0.0
    fp[-1] = 1.0
    return xp, fp


def apply_curve_piecewise_linear(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor):
    """
    x: torch tensor, assumed normalized to [0,1]
    xp, fp: knots from sample_monotone_curve
    returns: remapped x in [0,1]
    """
    # clamp input and ensure contiguous for torch.bucketize
    x = x.clamp(0.0, 1.0).contiguous()

    # bucketize: find segment index i such that xp[i] <= x < xp[i+1]
    # torch.bucketize returns index in [0..len(xp)]
    idx = torch.bucketize(x, xp) - 1
    idx = idx.clamp(0, xp.numel() - 2)

    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]

    # linear interpolation within the segment
    t = (x - x0) / (x1 - x0 + 1e-6)
    y = y0 + t * (y1 - y0)
    return y


@torch.no_grad()
def intensity_remap(x: torch.Tensor,
                    num_knots: int = 8,
                    strength: float = 0.25):
    """
    x: (B, C, ...) float tensor
    Assumption: x already normalized to [0,1] (nnUNet often gives roughly normalized patches)
    """
    device = x.device
    dtype = x.dtype
    xp, fp = sample_monotone_curve(num_knots=num_knots, strength=strength, device=device, dtype=dtype)
    return apply_curve_piecewise_linear(x, xp, fp)

def test_curve_properties():
    x = torch.rand(4, 1, 128, 128, 128).cuda()
    x_aug = intensity_remap(x, num_knots=8, strength=0.4)
    print(x_aug.shape)
    assert torch.isfinite(x_aug).all()
    assert x_aug.min() >= 0.0 - 1e-5
    assert x_aug.max() <= 1.0 + 1e-5

    # monotonicity check on 1D sample
    xp, fp = sample_monotone_curve(device=x.device)
    assert torch.all(fp[1:] >= fp[:-1])

if __name__ == "__main__":
    test_curve_properties()

