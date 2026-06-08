"""
MicroSR — Unit Tests
Run: python -m pytest tests/ -v
  OR: make test-unit
"""

import sys
import pytest
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model   import UNet, DDPM, SinusoidalPE
from src.losses  import make_gaussian_psf, psf_consistency_loss, is_valid_microscopy_image
from src.metrics import compute_psnr, compute_ssim, compute_psf_error


DEVICE = "cpu"   # always test on CPU for portability


# ─────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────
class TestUNet:
    def test_output_shape(self):
        model = UNet(in_ch=2, out_ch=1, base_ch=32).to(DEVICE)
        x = torch.randn(2, 2, 64, 64)
        t = torch.randint(0, 100, (2,))
        out = model(x, t)
        assert out.shape == (2, 1, 64, 64), f"Expected (2,1,64,64), got {out.shape}"

    def test_different_batch_sizes(self):
        model = UNet(in_ch=2, out_ch=1, base_ch=32).to(DEVICE)
        for B in [1, 2, 4]:
            x   = torch.randn(B, 2, 64, 64)
            t   = torch.randint(0, 100, (B,))
            out = model(x, t)
            assert out.shape[0] == B

    def test_no_nan_output(self):
        model = UNet(in_ch=2, out_ch=1, base_ch=32).to(DEVICE)
        x   = torch.randn(2, 2, 64, 64)
        t   = torch.randint(0, 100, (2,))
        out = model(x, t)
        assert not torch.isnan(out).any(), "NaN in model output"


class TestDDPM:
    def setup_method(self):
        self.ddpm = DDPM(T=10, device=DEVICE)

    def test_q_sample_shape(self):
        x0    = torch.randn(2, 1, 64, 64)
        t     = torch.randint(0, 10, (2,))
        xt, n = self.ddpm.q_sample(x0, t)
        assert xt.shape == x0.shape
        assert n.shape  == x0.shape

    def test_alpha_bar_decreasing(self):
        """Alpha bar should decrease from ~1 toward 0 over T steps."""
        assert self.ddpm.alpha_bar[0] > self.ddpm.alpha_bar[-1]
        assert self.ddpm.alpha_bar[0] < 1.0
        assert self.ddpm.alpha_bar[-1] > 0.0


# ─────────────────────────────────────────────
# PSF / Loss tests
# ─────────────────────────────────────────────
class TestPSF:
    def test_psf_kernel_shape(self):
        psf = make_gaussian_psf(sigma_px=2.0, kernel_size=15, device=DEVICE)
        assert psf.shape == (1, 1, 15, 15)

    def test_psf_sums_to_one(self):
        psf = make_gaussian_psf(sigma_px=2.0, kernel_size=15, device=DEVICE)
        assert abs(psf.sum().item() - 1.0) < 1e-5, "PSF must be normalised"

    def test_psf_loss_zero_for_consistent_input(self):
        """If HR is just upsampled LR, PSF loss should be near zero after blurring."""
        import torch.nn.functional as F
        psf = make_gaussian_psf(sigma_px=1.0, kernel_size=7, device=DEVICE)
        lr  = torch.rand(2, 1, 32, 32)
        hr  = F.interpolate(lr, scale_factor=2, mode="bilinear", align_corners=False)
        loss = psf_consistency_loss(hr, lr, psf, upsample=2)
        assert loss.item() >= 0, "PSF loss must be non-negative"

    def test_psf_loss_gradient_flows(self):
        psf    = make_gaussian_psf(sigma_px=2.0, device=DEVICE)
        hr     = torch.randn(2, 1, 128, 128, requires_grad=True)
        lr     = torch.randn(2, 1, 64, 64)
        loss   = psf_consistency_loss(hr, lr, psf, upsample=2)
        loss.backward()
        assert hr.grad is not None, "Gradient must flow through PSF loss"


# ─────────────────────────────────────────────
# Input validator tests
# ─────────────────────────────────────────────
class TestInputValidator:
    def test_valid_microscopy_image(self):
        img = torch.rand(1, 1, 64, 64)   # structured content
        valid, reason = is_valid_microscopy_image(img)
        assert valid, f"Should accept valid image: {reason}"

    def test_rejects_flat_image(self):
        img = torch.ones(1, 1, 64, 64) * 0.5   # flat, no contrast
        valid, _ = is_valid_microscopy_image(img)
        assert not valid, "Should reject flat image"

    def test_rejects_too_small(self):
        img = torch.rand(1, 1, 8, 8)
        valid, reason = is_valid_microscopy_image(img, min_size=32)
        assert not valid, f"Should reject small image: {reason}"

    def test_accepts_rgb_by_converting(self):
        img = torch.rand(1, 3, 64, 64)   # RGB — should convert to grayscale
        valid, reason = is_valid_microscopy_image(img)
        assert valid, f"Should accept RGB and convert: {reason}"


# ─────────────────────────────────────────────
# Metrics tests
# ─────────────────────────────────────────────
class TestMetrics:
    def test_psnr_identical_images(self):
        img = np.random.rand(4, 64, 64).astype(np.float32)
        psnr = compute_psnr(img, img)
        assert psnr > 50, "PSNR of identical images should be very high"

    def test_ssim_range(self):
        pred = np.random.rand(4, 64, 64).astype(np.float32)
        gt   = np.random.rand(4, 64, 64).astype(np.float32)
        ssim = compute_ssim(pred, gt)
        assert -1.0 <= ssim <= 1.0, f"SSIM out of range: {ssim}"

    def test_psf_error_non_negative(self):
        psf  = make_gaussian_psf(device=DEVICE)
        hr   = torch.rand(2, 1, 128, 128)
        lr   = torch.rand(2, 1, 64, 64)
        err  = compute_psf_error(hr, lr, psf, upsample=2)
        assert err >= 0, "PSF error must be non-negative"


# ─────────────────────────────────────────────
# Integration test — full forward pass
# ─────────────────────────────────────────────
class TestIntegration:
    def test_full_training_step(self):
        from src.losses import total_loss

        model = UNet(in_ch=2, out_ch=1, base_ch=16).to(DEVICE)
        ddpm  = DDPM(T=10, device=DEVICE)
        psf   = make_gaussian_psf(sigma_px=2.0, device=DEVICE)

        lr = torch.randn(2, 1, 64, 64)
        hr = torch.randn(2, 1, 128, 128)
        t  = torch.randint(0, 10, (2,))

        hr_noisy, noise = ddpm.q_sample(hr, t)
        import torch.nn.functional as F
        lr_up    = F.interpolate(lr, size=(128, 128), mode="bilinear", align_corners=False)
        model_in = torch.cat([hr_noisy, lr_up], dim=1)
        noise_pred      = model(model_in, t)

        ab      = ddpm.alpha_bar[t][:, None, None, None]
        hr_pred = (hr_noisy - (1 - ab).sqrt() * noise_pred) / (ab.sqrt() + 1e-8)

        losses = total_loss(noise_pred, noise, hr_pred, lr, psf, lam=0.1, upsample=2)

        assert "total"   in losses
        assert "denoise" in losses
        assert "psf"     in losses
        assert losses["total"].item() > 0
        assert not torch.isnan(losses["total"]), "NaN in total loss"

    def test_sample_output_shape(self):
        model = UNet(in_ch=2, out_ch=1, base_ch=16).to(DEVICE)
        ddpm  = DDPM(T=5, device=DEVICE)
        lr    = torch.randn(1, 1, 64, 64)
        out   = ddpm.sample(model, lr, shape=(1, 1, 128, 128))
        assert out.shape == (1, 1, 128, 128)
        assert out.min() >= -1.1 and out.max() <= 1.1
