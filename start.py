#!/usr/bin/env python3
"""
MicroSR — Starter Script
Run this first. It checks your environment, installs
everything, downloads a small dataset sample, and
walks you through the entire project step by step.

Usage:
    python start.py
"""

import sys
import os
import subprocess
import platform
from pathlib import Path


# ─────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════╗
║   🔬  MicroSR — Physics-Constrained Microscopy SR       ║
║       Final Year Project Setup Wizard                   ║
╚══════════════════════════════════════════════════════════╝
"""

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg):   print(f"  {RED}✗{RESET} {msg}")
def info(msg):  print(f"  {CYAN}→{RESET} {msg}")
def step(n, msg): print(f"\n{BOLD}[Step {n}]{RESET} {msg}")
def hr():       print("  " + "─" * 54)


def run(cmd: list, capture=False):
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True)
    return subprocess.run(cmd)


# ─────────────────────────────────────────────
# Step 1 — Python version
# ─────────────────────────────────────────────
def check_python():
    step(1, "Checking Python version")
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 9:
        ok(f"Python {major}.{minor} — compatible")
        return True
    else:
        err(f"Python {major}.{minor} found — need Python 3.9+")
        info("Download: https://www.python.org/downloads/")
        return False


# ─────────────────────────────────────────────
# Step 2 — GPU check
# ─────────────────────────────────────────────
def check_gpu():
    step(2, "Checking GPU availability")
    try:
        import torch
        if torch.cuda.is_available():
            name   = torch.cuda.get_device_name(0)
            vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"GPU: {name} | VRAM: {vram:.1f} GB")
            if vram < 6:
                warn("Less than 6GB VRAM — reduce batch_size to 2 in training")
            return True
        else:
            warn("No CUDA GPU found — will use CPU (training will be very slow)")
            warn("Recommended: use Kaggle (free P100 GPU) instead")
            return False
    except ImportError:
        warn("PyTorch not installed yet — will install in next step")
        return False


# ─────────────────────────────────────────────
# Step 3 — Install dependencies
# ─────────────────────────────────────────────
def install_deps():
    step(3, "Installing dependencies")
    req = Path("requirements.txt")
    if not req.exists():
        err("requirements.txt not found — are you in the project root?")
        sys.exit(1)

    info("Running: pip install -r requirements.txt")
    result = run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    if result.returncode == 0:
        ok("All dependencies installed")
    else:
        err("Dependency installation failed — check the error above")
        sys.exit(1)


# ─────────────────────────────────────────────
# Step 4 — Check project structure
# ─────────────────────────────────────────────
def check_structure():
    step(4, "Verifying project structure")
    required = [
        "src/model.py", "src/losses.py", "src/dataset.py",
        "src/metrics.py", "train.py", "test.py",
        "frontend/app.py", "scripts/download_data.py",
        "Makefile", "requirements.txt", "configs/default.yaml"
    ]
    all_ok = True
    for f in required:
        if Path(f).exists():
            ok(f)
        else:
            err(f"Missing: {f}")
            all_ok = False

    if not all_ok:
        err("Project structure incomplete — re-download the zip file")
        sys.exit(1)


# ─────────────────────────────────────────────
# Step 5 — Dataset
# ─────────────────────────────────────────────
def setup_dataset():
    step(5, "Dataset setup")
    data_dir = Path("data/BioSR/MTs")

    if data_dir.exists():
        lr_count = len(list((data_dir / "LR").glob("*.tif"))) if (data_dir / "LR").exists() else 0
        ok(f"BioSR/MTs found — {lr_count} LR files")
        return

    print(f"""
  BioSR dataset not found. You have two options:

  {CYAN}Option A — Auto download (recommended){RESET}
    python scripts/download_data.py --structures MTs ER
    (~2 GB, takes 5-15 minutes)

  {CYAN}Option B — Manual download{RESET}
    1. Go to: https://figshare.com/articles/dataset/BioSR/13264793
    2. Download MTs.zip and ER.zip
    3. Extract to: data/BioSR/
    """)

    choice = input("  Auto-download now? [y/N]: ").strip().lower()
    if choice == "y":
        info("Starting download...")
        result = run([sys.executable, "scripts/download_data.py",
                      "--structures", "MTs", "ER",
                      "--out_dir", "data/BioSR"])
        if result.returncode == 0:
            ok("Dataset downloaded successfully")
        else:
            warn("Download failed — try manual download (Option B above)")
    else:
        warn("Skipping download — run 'python scripts/download_data.py' when ready")


# ─────────────────────────────────────────────
# Step 6 — MLOps setup
# ─────────────────────────────────────────────
def setup_mlops():
    step(6, "MLOps setup (optional but recommended)")

    # W&B
    try:
        import wandb
        r = run(["wandb", "status"], capture=True)
        if "logged in" in r.stdout.lower():
            ok("Weights & Biases — already logged in")
        else:
            warn("W&B not logged in")
            choice = input("  Set up W&B now? (free at wandb.ai) [y/N]: ").strip().lower()
            if choice == "y":
                run(["wandb", "login"])
    except ImportError:
        warn("W&B not installed (should have been installed in step 3)")

    # MLflow
    try:
        import mlflow
        ok("MLflow — installed")
        info("View experiments: make mlflow-ui  (opens at localhost:5000)")
    except ImportError:
        warn("MLflow not installed")


# ─────────────────────────────────────────────
# Step 7 — Quick smoke test
# ─────────────────────────────────────────────
def smoke_test():
    step(7, "Running smoke test (verifies model builds correctly)")
    test_code = """
import torch
import sys
sys.path.insert(0, '.')
from src.model  import UNet, DDPM
from src.losses import make_gaussian_psf, total_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = UNet(in_ch=2, out_ch=1, base_ch=32).to(device)
ddpm   = DDPM(T=10, device=device)
psf    = make_gaussian_psf(sigma_px=2.0, device=device)

# Fake batch
lr = torch.randn(2, 1, 64,  64, device=device)
hr = torch.randn(2, 1, 128, 128, device=device)
t  = torch.randint(0, 10, (2,), device=device)

noisy, noise = ddpm.q_sample(hr, t)
pred = model(torch.cat([noisy, lr], dim=1), t)

ab      = ddpm.alpha_bar[t][:, None, None, None]
hr_pred = (noisy - (1-ab).sqrt() * pred) / ab.sqrt()
losses  = total_loss(pred, noise, hr_pred, lr, psf, lam=0.1, upsample=2)

print(f"Loss: {losses['total'].item():.4f} | PSF: {losses['psf'].item():.4f}")
print("SMOKE TEST PASSED")
"""
    result = run([sys.executable, "-c", test_code], capture=True)
    if "SMOKE TEST PASSED" in result.stdout:
        ok("Model builds and runs correctly")
        info(result.stdout.strip().split('\n')[0])
    else:
        err("Smoke test failed:")
        print(result.stderr[-500:] if result.stderr else "No error output")
        return False
    return True


# ─────────────────────────────────────────────
# Step 8 — Print next steps
# ─────────────────────────────────────────────
def print_next_steps(all_ok: bool):
    step(8, "Setup complete — next steps")
    hr()

    if all_ok:
        ok("Everything is ready!")
    else:
        warn("Setup completed with some warnings (see above)")

    print(f"""
  {BOLD}Quick commands:{RESET}

  {CYAN}Train the model:{RESET}
    make train
    — or —
    python train.py --data_root data/BioSR --epochs 200 --lam 0.1 --amp --use_wandb

  {CYAN}Fast test run (50 epochs):{RESET}
    make train-fast

  {CYAN}Evaluate after training:{RESET}
    make test

  {CYAN}Launch the web app:{RESET}
    make app

  {CYAN}Deploy to HuggingFace:{RESET}
    See DEPLOYMENT.md

  {CYAN}View all commands:{RESET}
    make help

  {BOLD}Training time estimate (8GB GPU):{RESET}
    200 epochs, 64×64, batch 4 → ~6-8 hours
    Use --amp flag for faster mixed-precision training
    """)
    hr()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print(BANNER)
    print(f"  System: {platform.system()} {platform.machine()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  CWD:    {Path.cwd()}\n")

    results = []
    results.append(check_python())
    results.append(check_gpu())
    install_deps()
    check_structure()
    setup_dataset()
    setup_mlops()
    all_ok = smoke_test()
    print_next_steps(all_ok)


if __name__ == "__main__":
    main()
