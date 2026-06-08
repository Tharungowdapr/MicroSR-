"""
MicroSR — Physics-Constrained Diffusion Model
UNet backbone + DDPM noise scheduler
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Sinusoidal time embedding
# ─────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ─────────────────────────────────────────────
# Residual block
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1  = nn.GroupNorm(8, in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2  = nn.GroupNorm(8, out_ch)
        self.drop   = nn.Dropout(dropout)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


# ─────────────────────────────────────────────
# Self-attention block
# ─────────────────────────────────────────────
class AttnBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv.unbind(1)
        scale = C ** -0.5
        attn  = torch.softmax(q.transpose(-1, -2) @ k * scale, dim=-1)
        h     = (attn @ v.transpose(-1, -2)).transpose(-1, -2)
        return x + self.proj(h.reshape(B, C, H, W))


# ─────────────────────────────────────────────
# UNet — conditional on LR image + time step
# input channels = 2 (noisy HR + LR condition)
# ─────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(
        self,
        in_ch:   int = 2,   # noisy HR (1ch) + LR condition (1ch)
        out_ch:  int = 1,
        base_ch: int = 64,
        t_dim:   int = 256,
    ):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        # Time embedding MLP
        self.t_emb = nn.Sequential(
            SinusoidalPE(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # Encoder
        self.enc_in = nn.Conv2d(in_ch, ch[0], 3, padding=1)
        self.enc1   = ResBlock(ch[0], ch[0], t_dim)
        self.down1  = nn.Conv2d(ch[0], ch[1], 4, 2, 1)   # 64→32
        self.enc2   = ResBlock(ch[1], ch[1], t_dim)
        self.down2  = nn.Conv2d(ch[1], ch[2], 4, 2, 1)   # 32→16
        self.enc3   = ResBlock(ch[2], ch[2], t_dim)
        self.down3  = nn.Conv2d(ch[2], ch[3], 4, 2, 1)   # 16→8

        # Bottleneck
        self.mid1 = ResBlock(ch[3], ch[3], t_dim)
        self.attn = AttnBlock(ch[3])
        self.mid2 = ResBlock(ch[3], ch[3], t_dim)

        # Decoder
        self.up3  = nn.ConvTranspose2d(ch[3], ch[2], 4, 2, 1)
        self.dec3 = ResBlock(ch[2] * 2, ch[2], t_dim)
        self.up2  = nn.ConvTranspose2d(ch[2], ch[1], 4, 2, 1)
        self.dec2 = ResBlock(ch[1] * 2, ch[1], t_dim)
        self.up1  = nn.ConvTranspose2d(ch[1], ch[0], 4, 2, 1)
        self.dec1 = ResBlock(ch[0] * 2, ch[0], t_dim)

        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out_conv = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 2, H, W]  — noisy HR concat with LR condition
        t : [B]            — diffusion timestep
        """
        t_emb = self.t_emb(t)

        e0 = self.enc_in(x)
        e1 = self.enc1(e0, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)

        m = self.mid1(self.down3(e3), t_emb)
        m = self.attn(m)
        m = self.mid2(m, t_emb)

        d3 = self.dec3(torch.cat([self.up3(m), e3], 1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1), t_emb)

        return self.out_conv(F.silu(self.out_norm(d1)))


# ─────────────────────────────────────────────
# DDPM scheduler
# ─────────────────────────────────────────────
class DDPM:
    def __init__(
        self,
        T:          int   = 1000,
        beta_start: float = 1e-4,
        beta_end:   float = 0.02,
        device:     str   = "cuda",
    ):
        self.T      = T
        self.device = device

        betas          = torch.linspace(beta_start, beta_end, T, device=device)
        alphas         = 1.0 - betas
        alpha_bar      = torch.cumprod(alphas, dim=0)

        self.betas     = betas
        self.alphas    = alphas
        self.alpha_bar = alpha_bar

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,
        noise: torch.Tensor | None = None,
    ):
        """Add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    @torch.no_grad()
    def sample(
        self,
        model:     UNet,
        lr_cond:   torch.Tensor,       # [B,1,H,W]
        shape:     tuple = (1, 1, 64, 64),
    ) -> torch.Tensor:
        """Full reverse diffusion — generate HR from noise given LR condition."""
        x = torch.randn(shape, device=self.device)
        # Upsample LR condition to match HR spatial size
        hr_h, hr_w = shape[2], shape[3]
        lr_up = torch.nn.functional.interpolate(
            lr_cond, size=(hr_h, hr_w), mode="bilinear", align_corners=False
        )
        model.eval()
        for t in reversed(range(self.T)):
            t_batch    = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            model_in   = torch.cat([x, lr_up], dim=1)
            noise_pred = model(model_in, t_batch)

            beta  = self.betas[t]
            alpha = self.alphas[t]
            ab    = self.alpha_bar[t]

            x = (1 / alpha.sqrt()) * (x - (beta / (1 - ab).sqrt()) * noise_pred)
            if t > 0:
                x = x + beta.sqrt() * torch.randn_like(x)
        return x.clamp(-1, 1)
