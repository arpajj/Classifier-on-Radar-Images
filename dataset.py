"""
dataset.py
==========
Handles:
  - Scanning all .dat files across the 7 dataset folders
  - Parsing labels from filenames  (K digit → class index)
  - Signal processing per file:
      * Range-Time  (MTI filtered)
      * Spectrogram (summed over range bins)
      * Range-Doppler
  - Resizing all 3 representations to (IMAGE_HEIGHT x IMAGE_WIDTH)
  - Stacking into a 3-channel tensor  [3, H, W]
  - PyTorch Dataset + DataLoader creation with 80/10/10 split
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import config
from preprocessing import compute_representations, resize_image

# ══════════════════════════════════════════════════════════════════════
# 1.  FILE DISCOVERY
# ══════════════════════════════════════════════════════════════════════

def find_all_dat_files(root: str, folders: list) -> List[Tuple[str, int]]:
    """
    Walk through all dataset subfolders and collect (filepath, label) pairs.
    Label is extracted from the first character of the filename.

    Returns
    -------
    samples : list of (filepath, label_index)
    """
    samples = []
    for folder in folders:
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            print(f"  [WARNING] Folder not found: {folder_path}")
            continue
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith('.dat'):
                continue
            # Filename convention: KPXXAYYRZ.dat
            # First character K = activity digit (1-6)
            k = fname[0]
            if k not in config.LABEL_MAP:
                print(f"  [WARNING] Unrecognised label '{k}' in {fname}, skipping.")
                continue
            label = config.LABEL_MAP[k]
            filepath = os.path.join(folder_path, fname)
            samples.append((filepath, label))

    print(f"  Found {len(samples)} .dat files across {len(folders)} folders.")
    return samples


# ══════════════════════════════════════════════════════════════════════
# 2.  SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════

# Radar preprocessing is implemented in preprocessing.py


# ══════════════════════════════════════════════════════════════════════
# 3.  PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════

class RadarActivityDataset(Dataset):
    """
    PyTorch Dataset for radar human activity classification.
    Each item is:
        x : FloatTensor  [3, IMAGE_HEIGHT, IMAGE_WIDTH]
                channel 0 → Spectrogram
                channel 1 → Range-Time
                channel 2 → Range-Doppler
        y : int  (class index 0-5)
    """

    def __init__(self, samples: List[Tuple[str, int]]):
        """
        Parameters
        ----------
        samples : list of (filepath, label) tuples
        """
        self.samples = samples
        self.H = config.IMAGE_HEIGHT
        self.W = config.IMAGE_WIDTH

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        spec, rt, rd = compute_representations(filepath)

        # Resize to fixed spatial dimensions
        spec = resize_image(spec, self.H, self.W)
        rt   = resize_image(rt,   self.H, self.W)
        rd   = resize_image(rd,   self.H, self.W)

        # Stack → (3, H, W) float32 tensor
        x = torch.tensor(np.stack([spec, rt, rd], axis=0), dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y


# ══════════════════════════════════════════════════════════════════════
# 4.  DATALOADER FACTORY
# ══════════════════════════════════════════════════════════════════════

def get_dataloaders(root: str = config.DATASET_ROOT, folders: list = config.DATASET_FOLDERS,
                    train_ratio: float = config.TRAIN_RATIO, val_ratio: float = config.VAL_RATIO,
                    batch_size: int = config.BATCH_SIZE, num_workers: int = config.NUM_WORKERS,
                    seed: int = config.RANDOM_SEED,) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.
    Split is stratified per class to handle class imbalance.

    Returns
    -------
    train_loader, val_loader, test_loader
    """

    # ── Collect all files ─────────────────────────────────────────
    all_samples = find_all_dat_files(root, folders)
    if len(all_samples) == 0:
        raise RuntimeError("No .dat files found — check DATASET_ROOT and DATASET_FOLDERS in config.py")

    # ── Stratified split ──────────────────────────────────────────
    # Group samples by class
    from collections import defaultdict
    import random
    random.seed(seed)

    class_buckets = defaultdict(list)
    for s in all_samples:
        class_buckets[s[1]].append(s)

    train_samples, val_samples, test_samples = [], [], []
    for cls, items in class_buckets.items():
        random.shuffle(items)
        n       = len(items)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        # remainder → test
        train_samples += items[:n_train]
        val_samples   += items[n_train:n_train + n_val]
        test_samples  += items[n_train + n_val:]

    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # ── Datasets ──────────────────────────────────────────────────
    train_ds = RadarActivityDataset(train_samples)
    val_ds   = RadarActivityDataset(val_samples)
    test_ds  = RadarActivityDataset(test_samples)

    # ── DataLoaders ───────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = True)
    test_loader = DataLoader(test_ds, batch_size  = batch_size, shuffle = False, num_workers = num_workers, pin_memory = True)

    return train_loader, val_loader, test_loader

# ══════════════════════════════════════════════════════════════════════
# 5.  QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Scanning dataset …")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Grab one batch and print shapes
    x, y = next(iter(train_loader))
    print(f"  Batch x shape : {x.shape}")   # (B, 3, 128, 128)
    print(f"  Batch y shape : {y.shape}")   # (B,)
    print(f"  Labels in batch: {y.tolist()}")
    print(f"  x min/max: {x.min():.3f} / {x.max():.3f}")
    print("dataset.py sanity check passed.")
