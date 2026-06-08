# MicroSR — Project Understanding & Codebase Map

This document serves as a persistent guide to the MicroSR codebase. It explains the project architecture, training workflows, dataset details, and MLOps integrations.

## 📂 Directory Structure

```
microsr/
├── docs/
│   └── understanding.md  ← This file (project explanation)
├── src/
│   ├── model.py          ← Neural networks (UNet and DDPM scheduler)
│   ├── losses.py         ← Physics loss (Gaussian PSF consistency) and input validator
│   ├── dataset.py        ← BioSR dataset loader and split manager
│   └── metrics.py        ← Evaluation metrics (PSNR, SSIM, PSF error, FRC)
├── scripts/
│   ├── download_data.py  ← Downloader script for Figshare BioSR dataset
│   ├── setup_env.sh      ← Environment creation & GPU verification helper
│   ├── download_dataset.sh ← Script to automate downloading of all structures
│   ├── run_training.sh   ← Wrapper to start training with dynamic resource calibration
│   └── run_app.sh        ← Gradio application launcher
├── configs/
│   └── default.yaml      ← Default hyperparameters
├── frontend/
│   └── app.py            ← Gradio web interface
├── tests/
│   └── test_all.py       ← Pytest suite verifying model, scheduler, and loss function
├── Makefile              ← Easy shortcut targets for train, test, app, and cleanup
├── requirements.txt      ← Python environment dependencies
├── start.py              ← Original setup wizard
├── train.py              ← Model training script
└── test.py               ← Model evaluation/ablation script
```

---

## 🔬 Core Components & Architecture

### 1. Neural Networks (`src/model.py`)
- **`UNet`**: A conditional UNet backbone containing:
  - **Sinusoidal Time Embeddings**: Embeds diffusion step $t$.
  - **Encoder/Decoder block**: Features Group Normalization, Residual blocks, and a self-attention bottleneck.
  - **Conditioning**: Concatenates the low-resolution (LR) image channel-wise to the noisy high-resolution (HR) image (input channels = 2, output channels = 1).
- **`DDPM`**: Noise scheduler with:
  - $T=1000$ diffusion timesteps.
  - Linear noise schedule ($\beta_{start}=10^{-4}$ to $\beta_{end}=0.02$).
  - Tweedie's formula approximation to estimate the clean HR image at each step.

### 2. Physics-Constrained Loss (`src/losses.py`)
- **`make_gaussian_psf`**: Generates a 2D Gaussian Point Spread Function (PSF) kernel representing microscope optical properties.
- **`psf_consistency_loss`**: Implements the physics constraint:
  $$Loss_{PSF} = \| (HR_{pred} * PSF)\downarrow - LR \|^2$$
  The generated HR image is convolved with the PSF, downsampled back to LR size, and compared to the original LR input.
- **`total_loss`**: Combines diffusion denoising loss (MSE of predicted vs true noise) and the PSF consistency loss:
  $$Loss_{Total} = Loss_{Denoise} + \lambda \cdot Loss_{PSF}$$

### 3. Dataset Pipeline (`src/dataset.py`)
- **`BioSRDataset`**: Loads aligned LR/HR `.tif` images from Figshare. Normalizes images to range $[-1, 1]$ for diffusion. Crops random $64\times 64$ patches (upsampled to $128\times 128$).
- **`make_dataloaders`**: Manages train/val dataloaders. Supports auto-detection of available structures and deterministic data splitting to avoid crashes if only a subset of structures is downloaded.

---

## 📊 MLOps Integration

- **Weights & Biases (W&B)**: Logs training loss, validation loss, PSNR, SSIM, and gradients. Run `wandb login` to connect.
- **MLflow**: Tracks experiment parameters, logs epoch metrics, and registers model checkpoints in local SQLite or file storage (`mlruns/`). Launch dashboard using `make mlflow-ui`.

---

## 🚀 Execution & Lifecycles

1. **Setup**: Run `./scripts/setup_env.sh` to construct the Python virtual environment and check GPU configuration.
2. **Download**: Run `./scripts/download_dataset.sh` to download all structures (or subset) from Figshare.
3. **Train**: Run `./scripts/run_training.sh` to kick off the training loop with safety checks.
4. **Test**: Run `make test` to evaluate metrics.
5. **Serve**: Run `make app` to start the web app.
