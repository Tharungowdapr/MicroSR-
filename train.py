"""
MicroSR — Training loop
DDPM + PSF constraint + W&B logging + MLflow + checkpointing
"""

import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# MLOps
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("W&B not installed — run: pip install wandb")

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("MLflow not installed — run: pip install mlflow")

from src.model   import UNet, DDPM
from src.losses  import make_gaussian_psf, total_loss
from src.dataset import make_dataloaders
from src.metrics import compute_metrics


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────
def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  [ckpt] Saved → {path}")


def load_checkpoint(path: str, model: UNet, optimizer, device: str) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  [ckpt] Resumed from epoch {ckpt['epoch']} — {path}")
    return ckpt["epoch"]


# ─────────────────────────────────────────────
# Validation step
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(model, ddpm, val_loader, psf_kernel, device, lam, upsample, max_batches=10):
    model.eval()
    all_metrics = []

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        # Sample HR from model
        B, _, H, W = lr.shape
        hr_gen = ddpm.sample(model, lr, shape=(B, 1, H * upsample, W * upsample))

        # Compute metrics
        m = compute_metrics(hr_gen, hr, lr, psf_kernel, upsample)
        all_metrics.append(m)

    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    model.train()
    return avg


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────
def train(cfg: dict):
    device  = cfg["device"]
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ──
    model  = UNet(in_ch=2, out_ch=1, base_ch=cfg["base_ch"]).to(device)
    ddpm   = DDPM(T=cfg["T"], device=device)
    psf    = make_gaussian_psf(sigma_px=cfg["psf_sigma"], device=device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {n_params/1e6:.1f}M")

    # ── Optimiser + scheduler ──
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    scaler = GradScaler(enabled=cfg["amp"])          # mixed precision

    # ── Data ──
    train_dl, val_dl = make_dataloaders(
        root        = cfg["data_root"],
        patch_size  = cfg["patch_size"],
        upsample    = cfg["upsample"],
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
        max_train   = cfg["max_train"],
        max_val     = cfg["max_val"],
    )

    # ── Resume from checkpoint ──
    start_epoch = 0
    if cfg.get("resume"):
        start_epoch = load_checkpoint(cfg["resume"], model, opt, device)

    # ── MLOps setup ──
    if HAS_WANDB and cfg["use_wandb"]:
        wandb.init(
            project = "microsr",
            name    = cfg["run_name"],
            config  = cfg,
        )
        wandb.watch(model, log="gradients", log_freq=100)

    if HAS_MLFLOW and cfg["use_mlflow"]:
        mlflow.set_experiment("microsr")
        mlflow.start_run(run_name=cfg["run_name"])
        mlflow.log_params(cfg)

    # ─────────────── TRAINING LOOP ───────────────
    print(f"\n[Train] Starting — {cfg['epochs']} epochs | λ={cfg['lam']} | device={device}")
    best_psf_err = float("inf")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_log = {"total": 0, "denoise": 0, "psf": 0}
        t0 = time.time()

        for step, batch in enumerate(train_dl):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            B  = lr.shape[0]

            # Sample random timestep
            t_step = torch.randint(0, ddpm.T, (B,), device=device)

            with autocast(enabled=cfg["amp"]):
                # Forward diffusion — add noise to HR
                noise    = torch.randn_like(hr)
                hr_noisy, _ = ddpm.q_sample(hr, t_step, noise)

                # Upsample LR to HR size for channel-wise concat
                lr_up    = torch.nn.functional.interpolate(
                    lr, size=hr.shape[2:], mode="bilinear", align_corners=False)

                # Predict noise — condition on LR
                model_in    = torch.cat([hr_noisy, lr_up], dim=1)
                noise_pred  = model(model_in, t_step)

                # Estimate clean HR (Tweedie approximation)
                ab      = ddpm.alpha_bar[t_step][:, None, None, None]
                hr_pred = (hr_noisy - (1 - ab).sqrt() * noise_pred) / (ab.sqrt() + 1e-8)

                # Downsample HR_pred for PSF comparison with LR
                losses = total_loss(
                    noise_pred, noise, hr_pred, lr,
                    psf, cfg["lam"], cfg["upsample"]
                )

            opt.zero_grad()
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            for k in epoch_log:
                epoch_log[k] += losses[k].item() if hasattr(losses[k], "item") else losses[k]

        sched.step()
        n_steps = len(train_dl)
        avg = {k: v / n_steps for k, v in epoch_log.items()}
        avg["lr"]  = sched.get_last_lr()[0]
        avg["epoch"] = epoch
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d} | loss={avg['total']:.4f} "
              f"denoise={avg['denoise']:.4f} psf={avg['psf']:.4f} "
              f"| {elapsed:.0f}s")

        # ── Validation every 10 epochs ──
        if epoch % cfg["val_every"] == 0:
            val_m = validate(model, ddpm, val_dl, psf, device,
                             cfg["lam"], cfg["upsample"])
            avg.update({f"val_{k}": v for k, v in val_m.items()})
            print(f"         Val | psnr={val_m.get('psnr', 0):.2f} "
                  f"ssim={val_m.get('ssim', 0):.4f} "
                  f"psf_err={val_m.get('psf_err', 0):.4f}")

            # Save best model
            if val_m.get("psf_err", float("inf")) < best_psf_err:
                best_psf_err = val_m["psf_err"]
                save_checkpoint(
                    {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": opt.state_dict(), "cfg": cfg},
                    str(run_dir / "best_model.pt")
                )

        # ── Periodic checkpoint ──
        if epoch % cfg["save_every"] == 0:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": opt.state_dict(), "cfg": cfg},
                str(run_dir / f"ckpt_ep{epoch:03d}.pt")
            )

        # ── MLOps logging ──
        if HAS_WANDB and cfg["use_wandb"]:
            wandb.log(avg)
        if HAS_MLFLOW and cfg["use_mlflow"]:
            mlflow.log_metrics(avg, step=epoch)

    # ── Final save ──
    final_path = str(run_dir / "final_model.pt")
    save_checkpoint(
        {"epoch": cfg["epochs"], "model": model.state_dict(),
         "optimizer": opt.state_dict(), "cfg": cfg},
        final_path
    )
    print(f"\n[Done] Final model → {final_path}")

    if HAS_WANDB and cfg["use_wandb"]:
        wandb.finish()
    if HAS_MLFLOW and cfg["use_mlflow"]:
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()

    return model


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroSR training")
    parser.add_argument("--data_root",   required=True,        help="Path to BioSR root")
    parser.add_argument("--run_dir",     default="runs/exp1",  help="Output directory")
    parser.add_argument("--run_name",    default="microsr-v1", help="Experiment name")
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--lam",         type=float, default=0.1,  help="PSF loss weight")
    parser.add_argument("--T",           type=int,   default=1000, help="Diffusion steps")
    parser.add_argument("--patch_size",  type=int,   default=64)
    parser.add_argument("--upsample",  type=int,   default=2)
    parser.add_argument("--base_ch",     type=int,   default=64)
    parser.add_argument("--psf_sigma",   type=float, default=2.0)
    parser.add_argument("--max_train",   type=int,   default=2000)
    parser.add_argument("--max_val",     type=int,   default=200)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--val_every",   type=int,   default=10)
    parser.add_argument("--save_every",  type=int,   default=20)
    parser.add_argument("--resume",      default=None,         help="Path to checkpoint")
    parser.add_argument("--amp",         action="store_true",  help="Mixed precision")
    parser.add_argument("--use_wandb",   action="store_true")
    parser.add_argument("--use_mlflow",  action="store_true")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(vars(args))
