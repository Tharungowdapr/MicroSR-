# 🔬 MicroSR — Physics-Constrained Microscopy Super-Resolution

> A diffusion model that enhances blurry microscope images while ensuring the output obeys real optical physics (PSF consistency constraint).  
> **B.Tech Final Year Project | 2025–26**

---

## 🚀 Quick Start

```bash
# Clone & run the app with our pre-trained model
git clone https://github.com/Tharungowdapr/MicroSR-.git
cd MicroSR-
pip install -r requirements.txt
python frontend/app.py --ckpt runs/microsr-gpu-v2/best_model.pt
# → Opens at http://localhost:7860
```

## 🧠 Model

| Component | Detail |
|-----------|--------|
| Backbone | Conditional UNet (21.1M params) |
| Scheduler | DDPM, T=200 steps |
| Loss | MSE denoising + λ × PSF consistency |
| Training | 200 epochs, batch 8, Adam, AMP |
| GPU | RTX 5060 8GB, ~40 min training |

## 📊 Performance

| Metric | GPU (200 epochs) | CPU (43 epochs) |
|--------|-----------------|-----------------|
| PSNR | **27.24 dB** | 8.57 dB |
| SSIM | **0.43** | 0.04 |
| PSF Error | **0.041** | 0.44 |
| FRC | **0.68** | 0.35 |

## 📦 Data

**BioSR dataset** — 2200+ paired LR/HR fluorescence microscopy images.  
Train: MTs (microtubules), ER (endoplasmic reticulum).  
Test: F-actin, CCPs (cross-structure generalization).

Download: `python scripts/download_data.py`

## 🏋️ Train from scratch

```bash
python train.py --data_root data/BioSR --device cuda --epochs 200 --batch_size 8 --amp --use_mlflow
```

## 📈 MLOps

- **MLflow**: `mlflow ui --port 5001` — full metrics & charts
- **DVC**: `dvc status` — data & model versioning

## 📁 Project Structure

```
microsr/
├── train.py              # Training
├── frontend/app.py       # Gradio web app
├── src/
│   ├── model.py          # UNet + DDPM
│   ├── losses.py         # PSF loss
│   ├── dataset.py        # Data loader
│   └── metrics.py        # PSNR, SSIM, PSF, FRC
├── scripts/download_data.py
├── configs/default.yaml
├── microsr_colab.ipynb   # Colab notebook
├── requirements.txt
└── Makefile
```

## 📚 References

- [DDPM: Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [BioSR Dataset](https://figshare.com/articles/dataset/BioSR/13264793)
- [SR3: Image Super-Resolution via Iterative Refinement](https://arxiv.org/abs/2104.07636)

## 🔗 GitHub

https://github.com/Tharungowdapr/MicroSR-.git

## License

MIT
