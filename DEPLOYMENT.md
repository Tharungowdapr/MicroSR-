# MicroSR — Deployment Guide

## Option A — HuggingFace Spaces (Recommended, Free)

### Step 1: Create a HuggingFace account
Go to https://huggingface.co and sign up (free).

### Step 2: Create a new Space
1. Go to https://huggingface.co/new-space
2. Name: `microsr`
3. SDK: **Gradio**
4. Visibility: Public
5. Hardware: **CPU Basic** (free) or **T4 Small** ($0.05/hr)
6. Click **Create Space**

### Step 3: Install HuggingFace CLI
```bash
pip install huggingface-hub
huggingface-cli login   # enter your HF token from hf.co/settings/tokens
```

### Step 4: Upload your model checkpoint
```bash
# Upload the trained model to HuggingFace Model Hub
python scripts/upload_model.py \
    --ckpt runs/exp1/best_model.pt \
    --repo your-username/microsr-model
```

### Step 5: Push the Space code
```bash
# Clone your Space repo
git clone https://huggingface.co/spaces/your-username/microsr
cd microsr

# Copy project files
cp -r /path/to/microsr/* .

# Edit frontend/app.py — update CKPT_PATH to load from HF Hub:
# from huggingface_hub import hf_hub_download
# ckpt = hf_hub_download("your-username/microsr-model", "best_model.pt")

# Push
git add .
git commit -m "Initial deploy"
git push
```

Your app will be live at: **https://huggingface.co/spaces/your-username/microsr**

---

## Option B — Local with Public URL (Quick Share)

```bash
python frontend/app.py --ckpt runs/exp1/best_model.pt --share
```

This creates a temporary public URL via Gradio's tunneling.
Valid for 72 hours — good for demos.

---

## Option C — Docker (Self-hosted)

### Dockerfile
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "frontend/app.py", "--ckpt", "runs/exp1/best_model.pt", "--port", "7860"]
```

### Build and run
```bash
docker build -t microsr .
docker run -p 7860:7860 --gpus all microsr
```

---

## HuggingFace Spaces — Required Files Checklist

Make sure these are in your Space repo root:

| File | Purpose |
|------|---------|
| `README_HF.md` → rename to `README.md` | Space metadata (YAML frontmatter) |
| `frontend/app.py` | Main Gradio app (set as `app_file` in README) |
| `src/model.py` | UNet + DDPM |
| `src/losses.py` | PSF constraint |
| `src/metrics.py` | Evaluation metrics |
| `requirements.txt` | Dependencies |
| `runs/exp1/best_model.pt` | Trained model weights |

---

## Environment Variables for HF Spaces

Set these in your Space settings (Settings → Variables):

| Variable | Value | Purpose |
|----------|-------|---------|
| `WANDB_API_KEY` | your key | W&B logging in production |
| `MODEL_CKPT` | path or HF repo | Override checkpoint path |

---

## Hardware Recommendations

| Hardware | Cost | Inference speed | Good for |
|----------|------|----------------|----------|
| CPU Basic | Free | ~30-60s/image | Demo, testing |
| T4 Small | $0.05/hr | ~3-5s/image | Real use |
| A10G Small | $0.15/hr | ~1-2s/image | High traffic |

For a final year project demo, **CPU Basic (free)** is sufficient.
Inference will be slow (~45s) but it works.

---

## Reducing Inference Time for CPU Deployment

In `frontend/app.py`, reduce diffusion steps for faster CPU inference:

```python
# For deployment: use fewer steps (50 instead of 1000)
DDPM_OBJ = DDPM(T=50, device=DEVICE)   # Much faster on CPU
```

This reduces quality slightly but makes the demo usable on free CPU tier.
