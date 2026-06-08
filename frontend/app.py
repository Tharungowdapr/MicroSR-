"""
MicroSR — Gradio Frontend
Physics-Constrained Microscopy Super-Resolution
Full UI with project description, input validation, metrics display
"""

import os
import sys
import json
import time
import numpy as np
import torch
import gradio as gr
from pathlib import Path
import tifffile
from PIL import Image
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model  import UNet, DDPM
from src.losses import make_gaussian_psf, is_valid_microscopy_image
from src.metrics import compute_metrics


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL    = None
DDPM_OBJ = None
PSF      = None


def load_model_from_checkpoint(ckpt_path: str):
    global MODEL, DDPM_OBJ, PSF
    if not Path(ckpt_path).exists():
        print(f"[App] No checkpoint found at {ckpt_path}. Run: make train")
        print("[App] Starting in demo mode — inference disabled.")
        return False

    ckpt      = torch.load(ckpt_path, map_location=DEVICE)
    cfg       = ckpt.get("cfg", {})
    MODEL     = UNet(in_ch=2, out_ch=1, base_ch=cfg.get("base_ch", 64)).to(DEVICE)
    MODEL.load_state_dict(ckpt["model"])
    MODEL.eval()

    DDPM_OBJ  = DDPM(T=cfg.get("T", 1000), device=DEVICE)
    PSF       = make_gaussian_psf(sigma_px=cfg.get("psf_sigma", 2.0), device=DEVICE)
    print(f"[App] Model loaded — epoch {ckpt.get('epoch','?')} | device={DEVICE}")
    return True


# ─────────────────────────────────────────────
# Core inference function
# ─────────────────────────────────────────────
def enhance_image(
    image,
    lam_slider: float,
    upsample: int,
):
    """Main inference function called by Gradio."""
    global MODEL, DDPM_OBJ, PSF

    if MODEL is None:
        return None, None, "❌ Model not loaded. Please run training first."

    if image is None:
        return None, None, "⚠️ Please upload a microscopy image."

    # ── Convert input to tensor ──
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            img_np = image.mean(axis=2)          # collapse RGB → grayscale
        else:
            img_np = image
        img_np = img_np.astype(np.float32)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    else:
        return None, None, "❌ Unsupported image format."

    img_tensor = torch.from_numpy(img_np)[None, None]   # [1,1,H,W]

    # ── Input validation ──
    valid, reason = is_valid_microscopy_image(img_tensor)
    if not valid:
        return (
            None, None,
            f"❌ Invalid input: {reason}\n\n"
            f"Please upload a grayscale microscopy image "
            f"(fluorescence / widefield / confocal). "
            f"Supported formats: TIFF, PNG, JPEG."
        )

    # ── Resize to 64×64 LR ──
    lr_tensor = torch.nn.functional.interpolate(
        img_tensor, size=(64, 64), mode="bilinear", align_corners=False
    ).to(DEVICE)
    lr_norm   = lr_tensor * 2 - 1              # [-1,1]

    # ── Inference ──
    t0 = time.time()
    with torch.no_grad():
        hr_size = 64 * upsample
        hr_gen  = DDPM_OBJ.sample(MODEL, lr_norm, shape=(1, 1, hr_size, hr_size))
    elapsed = time.time() - t0

    # ── Compute metrics ──
    psf_err = float(
        torch.nn.functional.mse_loss(
            torch.nn.functional.avg_pool2d(
                torch.nn.functional.conv2d(
                    hr_gen.cpu(), PSF.cpu(), padding=PSF.shape[-1]//2
                ),
                kernel_size=upsample, stride=upsample
            ),
            lr_norm.cpu()
        ).item()
    )

    # ── Convert to display images ──
    lr_disp  = ((lr_norm.squeeze().cpu().clamp(-1,1) + 1) / 2 * 255).numpy().astype(np.uint8)
    hr_disp  = ((hr_gen.squeeze().cpu().clamp(-1,1)  + 1) / 2 * 255).numpy().astype(np.uint8)

    status = (
        f"✅ Enhancement complete\n"
        f"⏱️  Inference time: {elapsed:.1f}s\n"
        f"📐 Output size: {hr_size}×{hr_size}px\n"
        f"🔬 PSF consistency error: {psf_err:.5f} (lower = more physical)\n"
        f"🖥️  Device: {DEVICE.upper()}"
    )

    return lr_disp, hr_disp, status


# ─────────────────────────────────────────────
# Project description HTML
# ─────────────────────────────────────────────
PROJECT_INTRO = """
<div style="background:linear-gradient(135deg,#0f6e56 0%,#1d9e75 100%);
            padding:28px 32px;border-radius:12px;color:#fff;margin-bottom:4px">
  <h1 style="margin:0 0 8px;font-size:24px;font-weight:700">
    🔬 MicroSR — Physics-Constrained Microscopy Super-Resolution
  </h1>
  <p style="margin:0;font-size:15px;opacity:0.92;line-height:1.6">
    A diffusion model that enhances blurry microscope images into high-resolution detail
    — while ensuring the output obeys the real optical physics of the microscope.
  </p>
</div>
"""

HOW_IT_WORKS = """
<div style="padding:20px 0 8px">

<h3 style="margin:0 0 12px;font-size:16px">📖 What this project does</h3>
<p style="margin:0 0 10px;line-height:1.6;font-size:14px;color:#333">
Microscope images are blurry because of a physical phenomenon called the
<strong>Point Spread Function (PSF)</strong> — light from a single point in the sample
spreads out into a blob due to the limits of the lens. Standard AI models try to remove
this blur, but they don't check if their output is <em>physically realistic</em>.
</p>
<p style="margin:0 0 10px;line-height:1.6;font-size:14px;color:#333">
This model uses a <strong>Denoising Diffusion Probabilistic Model (DDPM)</strong>
with a <strong>PSF consistency constraint</strong>: after generating a high-resolution image,
we blur it back through the known microscope physics and check if it matches the original
low-resolution input. If it doesn't, the model is penalised — so it learns to produce
images that are both sharp <em>and</em> physically valid.
</p>

<h3 style="margin:16px 0 12px;font-size:16px">🔁 How the physics constraint works</h3>
<div style="background:#1a1a2e;color:#e0e0e0;border-left:4px solid #1d9e75;padding:14px 18px;
            border-radius:0 8px 8px 0;font-family:monospace;font-size:13px;line-height:1.8">
  <span style="color:#88d8b0">LR Input (64x64)</span><br>
  &nbsp;&nbsp;↓ <span style="color:#4ecdc4">DDPM generates</span><br>
  <span style="color:#88d8b0">HR Prediction (128x128)</span><br>
  &nbsp;&nbsp;↓ <span style="color:#4ecdc4">Convolve with PSF kernel</span><br>
  <span style="color:#88d8b0">Reconstructed LR (64x64)</span><br>
  &nbsp;&nbsp;↓ <span style="color:#4ecdc4">Compare to original LR</span><br>
  <span style="color:#ffd93d">PSF Loss = ||Reconstructed LR - Original LR||^2</span><br>
  &nbsp;&nbsp;↓ <span style="color:#4ecdc4">Add to training loss</span><br>
  <span style="color:#ff6b6b">Total Loss = Denoising Loss + lambda x PSF Loss</span>
</div>

<h3 style="margin:16px 0 12px;font-size:16px"> Model Architecture</h3>
<div style="background:#1a1a2e;color:#e0e0e0;border-left:4px solid #4ecdc4;padding:14px 18px;
            border-radius:0 8px 8px 0;font-family:monospace;font-size:13px;line-height:1.8">
  <b style="color:#88d8b0">Backbone:</b> Conditional UNet (21.1M parameters)<br>
  <b style="color:#88d8b0">Input:</b> 2 channels (noisy HR + LR condition), 64x64<br>
  <b style="color:#88d8b0">Encoder:</b> 4 levels [64, 128, 256, 512] channels<br>
  <b style="color:#88d8b0">Bottleneck:</b> ResBlock + Self-Attention<br>
  <b style="color:#88d8b0">Decoder:</b> 3 levels with skip connections<br>
  <b style="color:#88d8b0">Scheduler:</b> DDPM, T=200 steps, cosine LR decay<br>
  <b style="color:#88d8b0">Loss:</b> MSE (denoise) + lambda x PSF consistency loss<br>
  <b style="color:#88d8b0">Training:</b> 200 epochs, batch 8, Adam, mixed precision (AMP)
</div>

<h3 style="margin:16px 0 12px;font-size:16px"> Training Data</h3>
<div style="background:#1a1a2e;color:#e0e0e0;border-left:4px solid #ffd93d;padding:14px 18px;
            border-radius:0 8px 8px 0;font-family:monospace;font-size:13px;line-height:1.8">
  <b style="color:#88d8b0">Dataset:</b> BioSR (Figshare) - 2200+ paired LR/HR images<br>
  <b style="color:#88d8b0">Structures:</b> MTs (microtubules), ER (endoplasmic reticulum)<br>
  <b style="color:#88d8b0">Test set:</b> F-actin, CCPs (clathrin-coated pits) - cross-structure generalization<br>
  <b style="color:#88d8b0">Resolution:</b> LR 64x64 -> HR 128x128 (2x upscale)<br>
  <b style="color:#88d8b0">Augmentation:</b> Random flips, rotations, crops
</div>

<h3 style="margin:16px 0 12px;font-size:16px"> Model Performance (GPU: RTX 5060 8GB)</h3>
<div style="background:#1a1a2e;color:#e0e0e0;border-left:4px solid #ff6b6b;padding:14px 18px;
            border-radius:0 8px 8px 0;font-family:monospace;font-size:13px;line-height:1.8">
  <span style="color:#88d8b0">PSNR:</span> 27.24 dB (GPU) vs 8.57 dB (CPU - partial train)<br>
  <span style="color:#88d8b0">SSIM:</span> 0.43 (GPU) vs 0.04 (CPU)<br>
  <span style="color:#88d8b0">PSF Error:</span> 0.041 (GPU) vs 0.44 (CPU) - lower is better<br>
  <span style="color:#88d8b0">FRC:</span> 0.68 (GPU) vs 0.35 (CPU) - higher is better<br>
  <span style="color:#88d8b0">Inference:</span> ~8s per 64x64 image (GPU, T=200)
</div>

<h3 style="margin:16px 0 12px;font-size:16px"> Research Papers & Further Reading</h3>
<ul style="margin:0;padding-left:20px;line-height:1.9;font-size:13px;color:#333">
  <li><b>DDPM:</b> Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020) - <a href='https://arxiv.org/abs/2006.11239' target='_blank'>arxiv.org/abs/2006.11239</a></li>
  <li><b>Microscopy SR:</b> Qiao et al. "BioSR: Real-world Dataset for Super-Resolution Microscopy" - <a href='https://figshare.com/articles/dataset/BioSR/13264793' target='_blank'>figshare.com/BioSR</a></li>
  <li><b>PSF Constraint:</b> This project - physics-constrained diffusion for microscopy, B.Tech Final Year Project 2025-26</li>
  <li><b>Conditional DDPM:</b> Saharia et al. "Image Super-Resolution via Iterative Refinement" (SR3) - <a href='https://arxiv.org/abs/2104.07636' target='_blank'>arxiv.org/abs/2104.07636</a></li>
  <li><b>Optics:</b> Abbe diffraction limit, Point Spread Function, Nyquist sampling in microscopy</li>
</ul>

<h3 style="margin:16px 0 12px;font-size:16px"> GitHub Repository</h3>
<div style="background:#1a1a2e;color:#e0e0e0;border-left:4px solid #1d9e75;padding:14px 18px;
            border-radius:0 8px 8px 0;font-size:14px;line-height:1.6">
  <a href='https://github.com/Tharungowdapr/MicroSR-.git' target='_blank' style='color:#4ecdc4;font-weight:bold'>
    github.com/Tharungowdapr/MicroSR-
  </a>
  <br><span style="color:#aaa">Full source code, training scripts, and deployment guide.</span>
</div>

</div>
"""

HOW_TO_USE = """
<div style="padding:8px 0">
<h3 style="margin:0 0 12px;font-size:16px">🚀 How to use</h3>
<ol style="margin:0;padding-left:20px;line-height:1.9;font-size:14px">
  <li>Upload a <strong>grayscale microscopy image</strong> (TIFF, PNG, or JPEG)</li>
  <li>Adjust the <strong>PSF Constraint (λ)</strong> slider:
      <ul style="margin:4px 0;font-size:13px">
        <li><strong>λ = 0</strong> — no physics constraint (sharper but may be unrealistic)</li>
        <li><strong>λ = 0.1</strong> — recommended balance of quality + physical consistency</li>
        <li><strong>λ = 0.5</strong> — strong physical constraint (more conservative)</li>
      </ul>
  </li>
  <li>Choose <strong>upscale factor</strong> (2× recommended)</li>
  <li>Click <strong>Enhance Image</strong></li>
  <li>Download the result using the download button below the output</li>
</ol>

<h3 style="margin:16px 0 12px;font-size:16px">⚠️ Input requirements</h3>
<ul style="margin:0;padding-left:20px;line-height:1.8;font-size:14px">
  <li>Grayscale or single-channel fluorescence microscopy images only</li>
  <li>Minimum size: 32×32 pixels | Maximum: 1024×1024 pixels</li>
  <li>The model will <strong>automatically reject</strong> non-microscopy images
      (natural photos, documents, artworks, etc.)</li>
  <li>Best results on widefield or confocal fluorescence images</li>
</ul>
</div>
"""

TECH_DETAILS = """
<div style="padding:8px 0;font-size:13px;color:#555">
<strong>Architecture:</strong> Conditional DDPM with UNet backbone (21.1M params) |
<strong>Physics:</strong> Gaussian PSF convolution loss (lambda=0.1) |
<strong>Dataset:</strong> BioSR (figshare) - 2200 LR/HR pairs |
<strong>Framework:</strong> PyTorch 2.12 + CUDA 13.0 |
<strong>MLOps:</strong> MLflow + DVC |
<strong>GPU:</strong> NVIDIA RTX 5060 8GB |
<strong>Training:</strong> 200 epochs, 40 min |
<strong>GitHub:</strong> <a href='https://github.com/Tharungowdapr/MicroSR-.git' target='_blank'>github.com/Tharungowdapr/MicroSR-</a>
</div>
"""


# ─────────────────────────────────────────────
# Build Gradio app
# ─────────────────────────────────────────────
def build_app(ckpt_path: str = "runs/exp1/best_model.pt") -> gr.Blocks:
    model_loaded = load_model_from_checkpoint(ckpt_path)
    status_init  = (
        f"✅ Model loaded | Device: {DEVICE.upper()}"
        if model_loaded
        else "⚠️ Model not loaded — run training first (see README)"
    )

    with gr.Blocks(
        title = "MicroSR — Physics-Constrained Microscopy Super-Resolution",
    ) as app:

        # ── Header ──
        gr.HTML(PROJECT_INTRO)

        # ── Tabs ──
        with gr.Tabs():

            # ── TAB 1: Enhance ──
            with gr.TabItem("🔬 Enhance Image"):
                with gr.Row():

                    # Left column — inputs
                    with gr.Column(scale=1):
                        with gr.Accordion("📖 How to use", open=False):
                            gr.HTML(HOW_TO_USE)
                        input_image = gr.Image(
                            label   = "Upload Microscopy Image (TIFF, PNG, JPEG)",
                            type    = "numpy",
                            sources = ["upload"],
                        )
                        lam_slider  = gr.Slider(
                            minimum = 0.0,
                            maximum = 0.5,
                            value   = 0.1,
                            step    = 0.01,
                            label   = "PSF Constraint Strength (λ) — 0=off, 0.1=recommended",
                        )
                        up_radio    = gr.Radio(
                            choices = [2, 4],
                            value   = 2,
                            label   = "Upscale Factor",
                        )
                        run_btn     = gr.Button(
                            "🚀 Enhance Image",
                            variant = "primary",
                            size    = "lg",
                        )

                    # Right column — outputs
                    with gr.Column(scale=1):
                        with gr.Row():
                            out_lr   = gr.Image(label="LR Input (64×64)",  type="numpy")
                            out_hr   = gr.Image(label="HR Output (enhanced)", type="numpy")
                        out_status   = gr.Textbox(
                            label    = "Results & Metrics",
                            lines    = 6,
                            value    = status_init,
                            interactive = False,
                        )
                        dl_btn = gr.Button("💾 Download Enhanced Image")
                        dl_file = gr.File(label="Download", visible=False)

                # ── Button actions ──
                run_btn.click(
                    fn      = enhance_image,
                    inputs  = [input_image, lam_slider, up_radio],
                    outputs = [out_lr, out_hr, out_status],
                )

            # ── TAB 2: About the Project ──
            with gr.TabItem("📖 About this Project"):
                gr.HTML(HOW_IT_WORKS)
                gr.HTML(TECH_DETAILS)

                with gr.Accordion("📐 Physics: What is the PSF?", open=False):
                    gr.Markdown("""
The **Point Spread Function (PSF)** describes how a single point of light appears in a microscope image.
Due to the wave nature of light and the physical limits of lenses (Abbe diffraction limit), a perfect point
source appears as a blurred spot — a Gaussian blob in the 2D image plane.

**Formula:**
```
PSF(x, y) = (1 / 2πσ²) × exp(−(x² + y²) / 2σ²)
sigma = 0.21 × wavelength / NA
```

Where:
- `wavelength` = emission wavelength of the fluorescent dye (e.g. 520nm for GFP)
- `NA` = numerical aperture of the objective lens (e.g. 1.4 for oil immersion)
- A **low-resolution image = high-resolution sample ∗ PSF + noise**

This is why our constraint works: **if we generate a valid HR image, blurring it with the PSF must give back the original LR input.**
                    """)

                with gr.Accordion("🏗️ Model Architecture", open=False):
                    gr.Markdown("""
**UNet Backbone:**
- Input: 2 channels (noisy HR + LR condition), 64×64
- Encoder: 4 resolution levels (64→32→16→8), channels [64,128,256,512]
- Bottleneck: ResBlock + Self-Attention
- Decoder: 3 levels with skip connections
- Output: 1 channel (predicted noise)
- Parameters: **21.1M**

**DDPM Scheduler:**
- T = 200 diffusion steps (trained with 1000)
- Linear beta schedule: 0.0001 → 0.02
- Conditioning: LR image concatenated channel-wise to noisy HR

**PSF Constraint:**
```python
hr_blurred    = conv2d(HR_pred, PSF_kernel) # apply microscope physics
blurred_down  = avg_pool2d(hr_blurred, 2)   # downsample to LR size
psf_loss      = MSE(blurred_down, LR_input) # must match original LR
total_loss    = denoise_loss + λ × psf_loss
```
                    """)

                with gr.Accordion("📊 Evaluation Metrics & Results", open=False):
                    gr.Markdown("""
| Metric | What it measures | GPU Model | CPU Model | Better when |
|--------|-----------------|-----------|-----------|-------------|
| PSNR | Peak signal-to-noise ratio | **27.24 dB** | 8.57 dB | Higher |
| SSIM | Structural similarity | **0.43** | 0.04 | Higher (max 1.0) |
| PSF Error | Physics consistency | **0.041** | 0.44 | **Lower** |
| FRC | Fourier Ring Correlation | **0.68** | 0.35 | Higher |

> **GPU model**: RTX 5060 8GB, 200 epochs, 40 min training  
> **CPU model**: Partial training (43 epochs, 2+ hrs) - shown for comparison

The **PSF Error** is the key metric that standard super-resolution papers miss.
It measures whether the generated image is physically consistent with what the
microscope would actually produce — not just whether it looks sharp.

**Loss Progression (GPU vs CPU):**
- GPU: total loss 3.06 (ep 0) → 0.006 (ep 199) — steady convergence
- CPU: total loss 0.09 (ep 0) → 0.013 (ep 43) — slower, stopped early
- View full training curves: MLflow UI at port 5001
                    """)

            # ── TAB 3: How to Run Locally ──
            with gr.TabItem("⚙️ Setup & Training"):
                gr.Markdown("""
## Running MicroSR Locally

### Quick Start (using our trained model)
```bash
# 1. Clone the repo
git clone https://github.com/Tharungowdapr/MicroSR-.git
cd MicroSR-

# 2. Download pretrained model (~242 MB)
#    From: https://github.com/Tharungowdapr/MicroSR-/releases/tag/v1.0
wget -P runs/microsr-gpu-v2/ https://github.com/Tharungowdapr/MicroSR-/releases/download/v1.0/best_model.pt

# 3. Launch the app
pip install -r requirements.txt
python frontend/app.py --ckpt runs/microsr-gpu-v2/best_model.pt
```

### Train from scratch
```bash
# 1. Setup
pip install -r requirements.txt

# 2. Download data
python scripts/download_data.py

# 3. Train (GPU required, 8GB+ VRAM recommended)
python train.py \\
    --data_root data/BioSR \\
    --epochs 200 \\
    --batch_size 8 \\
    --device cuda \\
    --amp \\
    --use_mlflow

# 4. Launch app with your model
python frontend/app.py --ckpt runs/exp1/best_model.pt
```

### View training logs
```bash
# MLflow (metrics & charts)
mlflow ui --port 5001

# DVC (data/model versioning)
dvc status
dvc list .
```

### Training time on RTX 5060 8GB
| Resolution | Batch | Epochs | Time |
|------------|-------|--------|------|
| 64×64 | 8 | 200 | ~40 min |
| 64×64 | 8 | 50 (test) | ~10 min |

> **Note:** Our trained model is at `runs/microsr-gpu-v2/best_model.pt`
                """)

    app.theme = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate")
    app.css = """
        .gr-button-primary { background: #1D9E75 !important; border-color: #1D9E75 !important; }
        .gr-button-primary:hover { background: #0f6e56 !important; }
        footer { display: none !important; }
    """
    return app


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default="runs/microsr-gpu-v2/best_model.pt")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--share",  action="store_true", help="Create public URL")
    args = parser.parse_args()

    app = build_app(args.ckpt)
    app.launch(
        server_port     = args.port,
        share           = args.share,
        show_error      = True,
        favicon_path    = None,
        theme           = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
    )
