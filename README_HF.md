---
title: MicroSR - Physics-Constrained Microscopy Super-Resolution
emoji: 🔬
colorFrom: green
colorTo: teal
sdk: gradio
sdk_version: "4.44.0"
app_file: frontend/app.py
pinned: false
license: mit
tags:
  - diffusion
  - super-resolution
  - microscopy
  - physics-informed
  - biology
  - pytorch
models:
  - microsr/ddpm-psf-constrained
datasets:
  - BioSR
---

# MicroSR — Physics-Constrained Microscopy Super-Resolution

A DDPM-based super-resolution model that enhances blurry microscopy images
while enforcing optical PSF physics consistency.

## Model Details
- **Architecture**: Conditional DDPM with UNet backbone (~28M params)
- **Training data**: BioSR dataset (microtubules + endoplasmic reticulum)
- **Physics constraint**: Gaussian PSF consistency loss
- **Input**: 64×64 grayscale microscopy image
- **Output**: 128×128 (2×) or 256×256 (4×) enhanced image

## Usage
Upload a grayscale fluorescence microscopy image and click Enhance.
