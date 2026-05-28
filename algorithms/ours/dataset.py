import os
import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class BSD400Dataset(Dataset):
    """Random 64×64 patches from BSD400 raw images (grayscale Y channel)."""

    def __init__(self, root: str, patch_size: int = 64, patches_per_image: int = 50):
        img_names = [
            f for f in os.listdir(root)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        if not img_names:
            raise RuntimeError(f'No images found in {root}')
        self.paths = [os.path.join(root, f) for f in img_names] * patches_per_image
        self.patch_size = patch_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]  # Y channel, (H, W)
        h, w = img.shape
        p = self.patch_size

        if h < p or w < p:
            img = cv2.resize(img, (max(w, p), max(h, p)))
            h, w = img.shape

        top  = np.random.randint(0, h - p + 1)
        left = np.random.randint(0, w - p + 1)
        patch = img[top:top+p, left:left+p].astype(np.float32) / 255.0

        if np.random.rand() > 0.5:
            patch = patch[:, ::-1].copy()
        if np.random.rand() > 0.5:
            patch = patch[::-1, :].copy()

        return torch.from_numpy(patch).unsqueeze(0)   # (1, p, p)


# ---------------------------------------------------------------------------
# Test-image utilities
# ---------------------------------------------------------------------------

def load_test_image(path: str) -> np.ndarray:
    """Load image as float32 [0,1] grayscale (Y channel), shape (H, W)."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    return img.astype(np.float32) / 255.0


def img_to_blocks(img: np.ndarray, block_size: int = 64):
    """Pad to multiple of block_size, split into non-overlapping blocks.
    Returns blocks (K, block_size^2), padded H, padded W, orig H, orig W."""
    h0, w0 = img.shape
    ph = math.ceil(h0 / block_size) * block_size
    pw = math.ceil(w0 / block_size) * block_size
    pad = np.zeros((ph, pw), dtype=np.float32)
    pad[:h0, :w0] = img

    blocks = []
    for r in range(0, ph, block_size):
        for c in range(0, pw, block_size):
            blocks.append(pad[r:r+block_size, c:c+block_size].reshape(-1))

    return np.stack(blocks, axis=0), ph, pw, h0, w0


def blocks_to_img(blocks: np.ndarray, ph: int, pw: int,
                  h0: int, w0: int, block_size: int = 64) -> np.ndarray:
    """Reassemble blocks into image and crop to original size."""
    rec = np.zeros((ph, pw), dtype=np.float32)
    idx = 0
    for r in range(0, ph, block_size):
        for c in range(0, pw, block_size):
            rec[r:r+block_size, c:c+block_size] = blocks[idx].reshape(block_size, block_size)
            idx += 1
    return np.clip(rec[:h0, :w0], 0.0, 1.0)
