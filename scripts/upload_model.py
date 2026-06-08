"""
MicroSR — Upload trained model to HuggingFace Hub
Run after training: python scripts/upload_model.py --ckpt runs/exp1/best_model.pt
"""

import argparse
import shutil
import json
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    HAS_HF = True
except ImportError:
    HAS_HF = False


def upload_model(args):
    if not HAS_HF:
        print("Install huggingface-hub: pip install huggingface-hub")
        return

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        print("Run training first: make train")
        return

    api  = HfApi()
    repo = args.repo

    print(f"[Upload] Creating/verifying repo: {repo}")
    create_repo(repo, repo_type="model", exist_ok=True, private=args.private)

    # Build upload folder
    upload_dir = Path("hf_upload_tmp")
    upload_dir.mkdir(exist_ok=True)

    # Copy checkpoint
    shutil.copy(ckpt, upload_dir / "best_model.pt")

    # Copy model card
    model_card = f"""---
language: en
license: mit
tags:
  - diffusion
  - super-resolution
  - microscopy
  - physics-informed
  - pytorch
---

# MicroSR — Physics-Constrained Microscopy Super-Resolution

DDPM model with PSF consistency constraint for fluorescence microscopy SR.

## Usage
```python
import torch
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download("{repo}", "best_model.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
```

## Training
- Dataset: BioSR (microtubules + ER)
- Architecture: Conditional DDPM UNet (~28M params)
- PSF constraint: λ = {args.lam}
- Input: 64×64 grayscale | Output: 128×128
"""
    (upload_dir / "README.md").write_text(model_card)

    # Write config
    config = {"lam": args.lam, "psf_sigma": args.psf_sigma,
              "base_ch": 64, "T": 1000, "upsample": 2}
    (upload_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"[Upload] Uploading to {repo}...")
    api.upload_folder(
        folder_path = str(upload_dir),
        repo_id     = repo,
        repo_type   = "model",
    )

    shutil.rmtree(upload_dir)
    print(f"[Upload] Done — https://huggingface.co/{repo}")
    print(f"[Upload] Now update frontend/app.py to load from HF Hub:")
    print(f"""
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download("{repo}", "best_model.pt")
    load_model_from_checkpoint(ckpt)
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload MicroSR model to HuggingFace")
    parser.add_argument("--ckpt",       default="runs/exp1/best_model.pt")
    parser.add_argument("--repo",       required=True, help="e.g. your-username/microsr-model")
    parser.add_argument("--lam",        type=float, default=0.1)
    parser.add_argument("--psf_sigma",  type=float, default=2.0)
    parser.add_argument("--private",    action="store_true")
    upload_model(parser.parse_args())
