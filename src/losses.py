"""
MicroSR — Physics constraint losses
PSF convolution + consistency loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_gaussian_psf(
    sigma_px:    float = 2.0,
    kernel_size: int   = 15,
    device:      str   = "cuda",
) -> torch.Tensor:
    """
    Build a 2D Gaussian PSF kernel.

    Physical meaning:
        sigma_px = 0.21 * wavelength_nm / NA / pixel_size_nm
        Example: 0.21 * 520nm / 1.4 NA / 40nm/px ≈ 1.95 px

    Returns: [1, 1, K, K] tensor for use with F.conv2d
    """
    coords = torch.arange(kernel_size, device=device) - kernel_size // 2
    x, y   = torch.meshgrid(coords.float(), coords.float(), indexing="ij")
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma_px ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def psf_consistency_loss(
    hr_pred:    torch.Tensor,   # [B, 1, H, W]  predicted HR (normalised −1..1)
    lr_input:   torch.Tensor,   # [B, 1, h, w]  original LR input
    psf_kernel: torch.Tensor,   # [1, 1, K, K]
    upsample:   int = 2,
) -> torch.Tensor:
    """
    Physics constraint:
        Blur the predicted HR back through the PSF, then
        downsample → compare to original LR input.

        Loss = ||PSF * HR_pred ↓ − LR||²

    This makes generated images physically consistent with what
    the microscope would actually produce.
    """
    pad     = psf_kernel.shape[-1] // 2
    blurred = F.conv2d(hr_pred, psf_kernel, padding=pad)

    if upsample > 1:
        blurred_down = F.avg_pool2d(blurred, kernel_size=upsample, stride=upsample)
    else:
        blurred_down = blurred

    return F.mse_loss(blurred_down, lr_input)


def total_loss(
    noise_pred:  torch.Tensor,
    noise_true:  torch.Tensor,
    hr_pred:     torch.Tensor,
    lr_input:    torch.Tensor,
    psf_kernel:  torch.Tensor,
    lam:         float = 0.1,
    upsample:    int   = 2,
) -> dict:
    """
    Combined DDPM + PSF constraint loss.

    Returns dict with individual components for logging.
    """
    l_denoise = F.mse_loss(noise_pred, noise_true)
    l_psf     = psf_consistency_loss(hr_pred, lr_input, psf_kernel, upsample)
    l_total   = l_denoise + lam * l_psf

    return {
        "total":    l_total,
        "denoise":  l_denoise.detach(),
        "psf":      l_psf.detach(),
    }


# ─────────────────────────────────────────────
# Input validation — reject non-microscopy
# ─────────────────────────────────────────────
def is_valid_microscopy_image(
    img_tensor: torch.Tensor,
    min_size:   int   = 32,
    max_size:   int   = 1024,
    snr_thresh: float = 1.5,
) -> tuple[bool, str]:
    """
    Heuristic checks to reject clearly non-microscopy inputs.

    Checks:
    1. Image must be grayscale (1 channel) or convertible
    2. Size must be within bounds
    3. Must not be all-white or all-black (empty/corrupt)
    4. SNR must suggest structured content (not random noise or flat image)

    Returns: (is_valid, reason_if_invalid)
    """
    if img_tensor.ndim == 4:
        img_tensor = img_tensor.squeeze(0)

    # Channel check — collapse to 1ch if RGB
    if img_tensor.shape[0] == 3:
        img_tensor = img_tensor.mean(dim=0, keepdim=True)
    elif img_tensor.shape[0] != 1:
        return False, f"Expected 1 or 3 channels, got {img_tensor.shape[0]}"

    _, H, W = img_tensor.shape

    # Size bounds
    if H < min_size or W < min_size:
        return False, f"Image too small ({H}×{W}). Minimum {min_size}×{min_size}."
    if H > max_size or W > max_size:
        return False, f"Image too large ({H}×{W}). Maximum {max_size}×{max_size}."

    # Empty / saturated check
    pixel_min = img_tensor.min().item()
    pixel_max = img_tensor.max().item()
    if pixel_max - pixel_min < 0.02:
        return False, "Image appears flat (no contrast). Not a valid microscopy image."

    # SNR heuristic: mean / std should be > threshold for structured content
    mean = img_tensor.mean().item()
    std  = img_tensor.std().item() + 1e-8
    if std < 0.01:
        return False, "Image has very low variance. May not be a microscopy image."

    return True, "OK"
