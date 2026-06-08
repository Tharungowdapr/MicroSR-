"""
MicroSR — BioSR Dataset Loader
Handles LR/HR TIFF pairs from BioSR dataset
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import tifffile


class BioSRDataset(Dataset):
    """
    BioSR paired LR/HR microscopy dataset.

    Directory structure expected:
        root/
            MTs/
                LR/  *.tif
                HR/  *.tif
            F-actin/
                LR/
                HR/
            ER/
                LR/
                HR/
            CCPs/
                LR/
                HR/

    Download: https://figshare.com/articles/dataset/BioSR/13264793
    """

    STRUCTURES = ["MTs", "F-actin", "ER", "CCPs"]

    def __init__(
        self,
        root:        str,
        structures:  list[str]  = None,
        patch_size:  int        = 64,
        upsample:    int        = 2,
        augment:     bool       = True,
        max_samples: int | None = None,
        split:       str        = "all",
        split_ratio: float      = 0.9,
    ):
        """
        root:        Path to BioSR root directory
        structures:  Which biological structures to include (default: all)
        patch_size:  LR patch size — HR will be patch_size * upsample
        upsample:    Upscale factor (2 = linear SIM, 4 = non-linear SIM)
        augment:     Random flips and rotations
        max_samples: Cap dataset size (useful for quick tests)
        split:       "all", "train", or "val"
        split_ratio: ratio for train/val splitting of each structure
        """
        self.root       = Path(root)
        self.structures = structures or self.STRUCTURES
        self.patch_size = patch_size
        self.hr_size    = patch_size * upsample
        self.augment    = augment
        self.split       = split
        self.split_ratio = split_ratio

        self.pairs = []          # list of (lr_path, hr_path)
        self._build_pairs()

        if max_samples:
            random.shuffle(self.pairs)
            self.pairs = self.pairs[:max_samples]

        print(f"[BioSRDataset] {len(self.pairs)} pairs | "
              f"structures={self.structures} | patch={patch_size}→{self.hr_size} | split={self.split}")

    def _build_pairs(self):
        for struct in self.structures:
            lr_dir = self.root / struct / "LR"
            hr_dir = self.root / struct / "HR"
            if not lr_dir.exists():
                print(f"  Warning: {lr_dir} not found, skipping.")
                continue
            lr_files = sorted(lr_dir.glob("*.tif")) + sorted(lr_dir.glob("*.tiff"))
            
            struct_pairs = []
            for lr_path in lr_files:
                hr_path = hr_dir / lr_path.name
                if hr_path.exists():
                    struct_pairs.append((lr_path, hr_path))
            
            if len(struct_pairs) > 0:
                split_idx = int(len(struct_pairs) * self.split_ratio)
                if self.split == "train":
                    struct_pairs = struct_pairs[:split_idx]
                elif self.split == "val":
                    struct_pairs = struct_pairs[split_idx:]
            
            self.pairs.extend(struct_pairs)

    def _load_tiff(self, path: Path) -> np.ndarray:
        """Load TIFF, normalise to float32 [0,1]."""
        img = tifffile.imread(str(path)).astype(np.float32)
        if img.ndim == 3:
            img = img[0]                  # take first channel if multi-channel
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img

    def _random_crop(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop aligned LR / HR patches."""
        h, w   = lr.shape
        th, tw = self.patch_size, self.patch_size
        if h < th or w < tw:
            lr = np.pad(lr, ((0, max(0, th-h)), (0, max(0, tw-w))), mode="reflect")
            hr = np.pad(hr, ((0, max(0, self.hr_size-hr.shape[0])),
                             (0, max(0, self.hr_size-hr.shape[1]))), mode="reflect")
            h, w = lr.shape
        top  = random.randint(0, h - th)
        left = random.randint(0, w - tw)
        # Corresponding HR crop
        s    = self.hr_size // self.patch_size
        lr_c = lr[top:top+th, left:left+tw]
        hr_c = hr[top*s:(top+th)*s, left*s:(left+tw)*s]
        return lr_c, hr_c

    def _augment(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Paired flip + rotation."""
        if random.random() < 0.5:
            lr = np.fliplr(lr).copy()
            hr = np.fliplr(hr).copy()
        if random.random() < 0.5:
            lr = np.flipud(lr).copy()
            hr = np.flipud(hr).copy()
        k  = random.choice([0, 1, 2, 3])
        lr = np.rot90(lr, k).copy()
        hr = np.rot90(hr, k).copy()
        return lr, hr

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        lr_path, hr_path = self.pairs[idx]
        lr = self._load_tiff(lr_path)
        hr = self._load_tiff(hr_path)

        lr, hr = self._random_crop(lr, hr)
        if self.augment:
            lr, hr = self._augment(lr, hr)

        # Scale to [-1, 1] for DDPM
        lr_t = torch.from_numpy(lr)[None] * 2 - 1    # [1, H, W]
        hr_t = torch.from_numpy(hr)[None] * 2 - 1    # [1, H*s, W*s]

        return {"lr": lr_t, "hr": hr_t, "lr_path": str(lr_path)}


def make_dataloaders(
    root:         str,
    train_struct: list[str] = None,
    val_struct:   list[str] = None,
    patch_size:   int       = 64,
    upsample:     int       = 2,
    batch_size:   int       = 4,
    num_workers:  int       = 2,
    max_train:    int       = 2000,
    max_val:      int       = 200,
    split_ratio:  float     = 0.9,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders.

    Auto-detects structure availability. If the designated val_struct is not present,
    it falls back to using the training structures but partitions them with a split_ratio (e.g. 90/10)
    so that validation can always proceed without empty set crashes.
    """
    # Auto-detect which structures are actually present on disk
    root_path = Path(root)
    available_structures = []
    for s in ["MTs", "F-actin", "ER", "CCPs"]:
        if (root_path / s / "LR").exists():
            available_structures.append(s)

    # Resolve train structures
    train_struct = train_struct or ["MTs", "ER"]
    train_struct = [s for s in train_struct if s in available_structures]
    if not train_struct:
        # Fall back to any available structures
        train_struct = available_structures if available_structures else ["MTs"]

    # Resolve val structures
    val_struct = val_struct or ["F-actin", "CCPs"]
    val_struct = [s for s in val_struct if s in available_structures]

    # If the requested validation structures are not present, split from the train structures
    if not val_struct:
        print(f"[DataLoader] Validation structures not found. Splitting from train structures: {train_struct}")
        train_ds = BioSRDataset(root, train_struct, patch_size, upsample,
                                augment=True, max_samples=max_train, split="train", split_ratio=split_ratio)
        val_ds   = BioSRDataset(root, train_struct, patch_size, upsample,
                                augment=False, max_samples=max_val, split="val", split_ratio=split_ratio)
    else:
        print(f"[DataLoader] Using cross-structures for validation: {val_struct}")
        train_ds = BioSRDataset(root, train_struct, patch_size, upsample,
                                augment=True, max_samples=max_train, split="all")
        val_ds   = BioSRDataset(root, val_struct, patch_size, upsample,
                                augment=False, max_samples=max_val, split="all")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)

    return train_dl, val_dl
