# fusion.py
import torch

@torch.no_grad()
def normalize_saliency(sal: torch.Tensor, eps: float = 1e-6):
    """
    Normalize saliency to [0,1] per-sample (robust).
    sal: (B,1,...)
    """
    B = sal.shape[0]
    flat = sal.view(B, -1)
    s_min = flat.min(dim=1, keepdim=True).values
    s_max = flat.max(dim=1, keepdim=True).values
    norm = (flat - s_min) / (s_max - s_min + eps)
    return norm.view_as(sal).clamp(0.0, 1.0)


@torch.no_grad()
def sbf_fuse(
    x: torch.Tensor,          
    x_gla: torch.Tensor,
    x_lla: torch.Tensor,
    sal: torch.Tensor,
    mode: str = "soft",
    threshold: float = 0.5,
    temperature: float = 1.0,
):
    """
    Saliency-Balancing Fusion (paper-aligned, with explicit inverse saliency).

    x: original input image
    """
    assert x_gla.shape == x_lla.shape == x.shape
    assert sal.shape[0] == x.shape[0] and sal.shape[1] == 1

    # normalize saliency to [0,1]
    w = normalize_saliency(sal)      # saliency
    inv_w = 1.0 - w                  # inverse saliency

    # ---------- key addition: gate LLA ----------
    # high saliency -> suppress LLA
    inv_w_b = inv_w.expand_as(x)
    x_lla_gated = inv_w_b * x_lla + (1.0 - inv_w_b) * x
    # -------------------------------------------

    if mode == "soft":
        if temperature is not None and temperature != 1.0:
            w = torch.sigmoid((w - 0.5) * float(temperature))

        w_b = w.expand_as(x)
        out = w_b * x_gla + (1.0 - w_b) * x_lla_gated

    elif mode == "hard":
        mask = (w > threshold).expand_as(x)
        out = torch.where(mask, x_gla, x_lla_gated)

    else:
        raise ValueError(f"Unknown fusion mode: {mode}")

    return out


def test_sbf_fuse():
    x = torch.rand(4, 1, 128, 128, 128)
    x_gla = torch.rand(4, 1, 128, 128, 128)
    x_lla = torch.rand(4, 1, 128, 128, 128)
    sal = torch.rand(4, 1, 128, 128, 128)
    out = sbf_fuse(x, x_gla, x_lla, sal)
    print(out.shape)
    assert torch.isfinite(out).all()
    assert out.min() >= 0.0 - 1e-5
    assert out.max() <= 1.0 + 1e-5

if __name__ == "__main__":
    test_sbf_fuse()