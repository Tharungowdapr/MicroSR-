"""
Generate synthetic BioSR-compatible LR/HR microscopy image pairs.
Creates realistic fluorescence microscopy data using Gaussian spots + noise.
"""
import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import gaussian_filter
import argparse

RNG = np.random.default_rng(42)

def make_fluorescence_image(size=256, n_spots=80, sigma_psf=2.0):
    """Simulate a fluorescence microscopy image with Gaussian spots."""
    img = np.zeros((size, size), dtype=np.float32)
    for _ in range(n_spots):
        x = RNG.integers(10, size - 10)
        y = RNG.integers(10, size - 10)
        intensity = RNG.uniform(0.5, 1.0)
        patch_size = 12
        xs = np.arange(max(0, x-patch_size), min(size, x+patch_size))
        ys = np.arange(max(0, y-patch_size), min(size, y+patch_size))
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        img[xx, yy] += intensity * np.exp(-((xx-x)**2 + (yy-y)**2) / (2*sigma_psf**2))
    # Add Poisson noise
    img = np.clip(img, 0, None)
    img = RNG.poisson(img * 100).astype(np.float32) / 100.0
    img = np.clip(img / (img.max() + 1e-8), 0, 1)
    return img

def make_pair(hr_size=128, upsample=2, psf_sigma=2.0):
    hr = make_fluorescence_image(size=hr_size, sigma_psf=1.0)
    # LR = HR blurred by PSF + downsampled
    lr_blurred = gaussian_filter(hr, sigma=psf_sigma)
    lr_size = hr_size // upsample
    lr = lr_blurred[::upsample, ::upsample]
    assert lr.shape == (lr_size, lr_size)
    return lr.astype(np.float32), hr.astype(np.float32)

def generate(out_dir, structures, n_train=500, n_val=50, upsample=2):
    out_dir = Path(out_dir)
    for struct in structures:
        for split, n in [("train", n_train), ("val", n_val)]:
            lr_dir = out_dir / struct / "LR"
            hr_dir = out_dir / struct / "HR"
            lr_dir.mkdir(parents=True, exist_ok=True)
            hr_dir.mkdir(parents=True, exist_ok=True)

        total = n_train + n_val
        print(f"  Generating {total} pairs for {struct}...")
        for i in range(total):
            lr, hr = make_pair(hr_size=128, upsample=upsample)
            fname = f"{struct}_{i:04d}.tif"
            tifffile.imwrite(str(out_dir / struct / "LR" / fname), lr)
            tifffile.imwrite(str(out_dir / struct / "HR" / fname), hr)
            if (i+1) % 100 == 0:
                print(f"    {i+1}/{total}")
        print(f"  ✓ {struct}: {total} pairs written")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/BioSR")
    parser.add_argument("--structures", nargs="+", default=["MTs", "ER", "F-actin", "CCPs"])
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_val", type=int, default=50)
    args = parser.parse_args()

    print(f"[Synthetic] Generating BioSR-compatible data in {args.out_dir}/")
    generate(args.out_dir, args.structures, args.n_train, args.n_val)
    print("\n[Synthetic] Done! Run: make train")
