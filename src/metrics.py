"""
MicroSR — Evaluation Metrics
PSNR, SSIM, LPIPS, PSF consistency, Fourier Ring Correlation
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """[B,1,H,W] tensor in [-1,1] → [B,H,W] numpy in [0,1]"""
    return ((t.detach().cpu().clamp(-1, 1) + 1) / 2).squeeze(1).numpy()


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    scores = [psnr_fn(g, p, data_range=1.0) for g, p in zip(gt, pred)]
    return float(np.mean(scores))


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    scores = [ssim_fn(g, p, data_range=1.0) for g, p in zip(gt, pred)]
    return float(np.mean(scores))


def compute_psf_error(
    hr_pred:    torch.Tensor,
    lr_input:   torch.Tensor,
    psf_kernel: torch.Tensor,
    upsample:   int = 2,
) -> float:
    """Physical consistency error — lower is better."""
    with torch.no_grad():
        pad     = psf_kernel.shape[-1] // 2
        blurred = F.conv2d(hr_pred.cpu(), psf_kernel.cpu(), padding=pad)
        if upsample > 1:
            blurred = F.avg_pool2d(blurred, upsample, upsample)
        err = F.mse_loss(blurred, lr_input.cpu()).item()
    return err


def compute_fourier_ring_correlation(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Fourier Ring Correlation — measures resolution improvement.
    Higher = better spectral agreement with ground truth.
    """
    scores = []
    for p, g in zip(pred, gt):
        fp = np.fft.fftshift(np.fft.fft2(p))
        fg = np.fft.fftshift(np.fft.fft2(g))
        H, W   = p.shape
        cy, cx = H // 2, W // 2
        radii  = np.sqrt((np.arange(H)[:, None] - cy) ** 2 +
                         (np.arange(W)[None, :] - cx) ** 2).astype(int)
        frc_sum = 0.0
        n_rings = min(cy, cx)
        for r in range(1, n_rings):
            mask    = radii == r
            if mask.sum() == 0:
                continue
            num     = np.abs(np.sum(fp[mask] * np.conj(fg[mask])))
            den     = np.sqrt(np.sum(np.abs(fp[mask]) ** 2) *
                              np.sum(np.abs(fg[mask]) ** 2)) + 1e-8
            frc_sum += num / den
        scores.append(frc_sum / n_rings)
    return float(np.mean(scores))


def compute_metrics(
    hr_pred:    torch.Tensor,
    hr_gt:      torch.Tensor,
    lr_input:   torch.Tensor,
    psf_kernel: torch.Tensor,
    upsample:   int = 2,
) -> dict:
    """Compute all metrics. Returns dict of floats."""
    pred_np = tensor_to_numpy(hr_pred)
    gt_np   = tensor_to_numpy(hr_gt)

    return {
        "psnr":    compute_psnr(pred_np, gt_np),
        "ssim":    compute_ssim(pred_np, gt_np),
        "psf_err": compute_psf_error(hr_pred, lr_input, psf_kernel, upsample),
        "frc":     compute_fourier_ring_correlation(pred_np, gt_np),
    }
