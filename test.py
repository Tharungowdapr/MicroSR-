"""
MicroSR — Test / Evaluate a trained model
Runs all 4 experiments from the paper and saves results
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model   import UNet, DDPM
from src.losses  import make_gaussian_psf
from src.dataset import make_dataloaders, BioSRDataset
from src.metrics import compute_metrics
from torch.utils.data import DataLoader


def load_model(ckpt_path: str, device: str) -> tuple[UNet, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get("cfg", {})
    model = UNet(
        in_ch   = 2,
        out_ch  = 1,
        base_ch = cfg.get("base_ch", 64),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[Test] Loaded model from epoch {ckpt.get('epoch', '?')} — {ckpt_path}")
    return model, cfg


@torch.no_grad()
def run_evaluation(model, ddpm, loader, psf, device, upsample, n_batches=20):
    all_metrics = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        B  = lr.shape[0]
        hr_gen = ddpm.sample(model, lr, shape=(B, 1,
                             lr.shape[2] * upsample,
                             lr.shape[3] * upsample))
        m = compute_metrics(hr_gen, hr, lr, psf, upsample)
        all_metrics.append(m)
    return {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}


def save_visual_results(model, ddpm, loader, device, upsample, out_dir, n=4):
    """Save side-by-side LR / HR_gt / HR_pred comparison images."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(loader))
    lr    = batch["lr"][:n].to(device)
    hr    = batch["hr"][:n].to(device)
    hr_gen = ddpm.sample(model, lr, shape=(n, 1,
                         lr.shape[2] * upsample,
                         lr.shape[3] * upsample))

    def to_np(t):
        return ((t.cpu().clamp(-1,1) + 1) / 2).squeeze(1).numpy()

    lr_np, hr_np, gen_np = to_np(lr), to_np(hr), to_np(hr_gen)

    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
    titles = ["LR Input", "HR Ground Truth", "HR Generated (PSF-constrained)"]
    for i in range(n):
        for j, (img, title) in enumerate(zip([lr_np[i], hr_np[i], gen_np[i]], titles)):
            axes[i][j].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[i][j].set_title(title if i == 0 else "", fontsize=9)
            axes[i][j].axis("off")
    plt.tight_layout()
    out_path = out_dir / "visual_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Test] Visual results → {out_path}")


def plot_lambda_ablation(results: dict, out_dir: str):
    """Plot PSF error vs PSNR across lambda values."""
    lambdas  = sorted(results.keys())
    psf_errs = [results[l]["psf_err"] for l in lambdas]
    psnrs    = [results[l]["psnr"]    for l in lambdas]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(lambdas, psf_errs, "o-", color="#BA7517", label="PSF Error (↓ better)")
    ax2.plot(lambdas, psnrs,    "s-", color="#1D9E75", label="PSNR (↑ better)")
    ax1.set_xlabel("Lambda (λ) — PSF constraint weight")
    ax1.set_ylabel("PSF Consistency Error", color="#BA7517")
    ax2.set_ylabel("PSNR (dB)", color="#1D9E75")
    ax1.set_title("Pareto Curve: Image Quality vs Physical Consistency")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    plt.tight_layout()
    out = Path(out_dir) / "lambda_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Test] Lambda ablation plot → {out}")


def main(args):
    device  = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.ckpt, device)
    upsample   = cfg.get("upsample", 2)
    ddpm       = DDPM(T=cfg.get("T", 1000), device=device)
    psf        = make_gaussian_psf(sigma_px=cfg.get("psf_sigma", 2.0), device=device)

    # ── Experiment 1: In-domain evaluation ──
    print("\n[Exp 1] In-domain evaluation (MTs + ER)...")
    _, val_dl = make_dataloaders(
        root=args.data_root, patch_size=64,
        upsample=upsample, batch_size=4,
    )
    metrics = run_evaluation(model, ddpm, val_dl, psf, device, upsample)
    print(f"  Results: {json.dumps(metrics, indent=2)}")

    # ── Experiment 2: Cross-structure ──
    print("\n[Exp 2] Cross-structure generalisation (F-actin + CCPs)...")
    cross_ds = BioSRDataset(args.data_root, ["F-actin", "CCPs"],
                            patch_size=64, upsample=upsample, augment=False)
    cross_dl = DataLoader(cross_ds, batch_size=4, shuffle=False)
    cross_m  = run_evaluation(model, ddpm, cross_dl, psf, device, upsample)
    print(f"  Results: {json.dumps(cross_m, indent=2)}")

    # ── Save metrics ──
    all_results = {"in_domain": metrics, "cross_structure": cross_m}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Test] Metrics saved → {out_dir / 'metrics.json'}")

    # ── Visual results ──
    print("\n[Test] Generating visual results...")
    save_visual_results(model, ddpm, val_dl, device, upsample, out_dir)

    print("\n[Test] Complete. Results in:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroSR evaluation")
    parser.add_argument("--ckpt",      required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", required=True, help="Path to BioSR root")
    parser.add_argument("--out_dir",   default="test_results")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
