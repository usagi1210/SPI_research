import math
import numpy as np
from skimage.metrics import structural_similarity as ssim


BLOCK_SIZE = 33


def imread_cs(img_gray: np.ndarray):
    """Pad a grayscale image so dimensions are multiples of BLOCK_SIZE."""
    row, col = img_gray.shape
    row_pad = BLOCK_SIZE - np.mod(row, BLOCK_SIZE)
    col_pad = BLOCK_SIZE - np.mod(col, BLOCK_SIZE)
    Ipad = np.concatenate([img_gray, np.zeros([row, col_pad])], axis=1)
    Ipad = np.concatenate([Ipad,    np.zeros([row_pad, col + col_pad])], axis=0)
    return img_gray, row, col, Ipad, *Ipad.shape


def img2col(Ipad: np.ndarray) -> np.ndarray:
    """Extract non-overlapping 33×33 blocks, return (num_blocks, 1089)."""
    row, col = Ipad.shape
    blocks = []
    for x in range(0, row - BLOCK_SIZE + 1, BLOCK_SIZE):
        for y in range(0, col - BLOCK_SIZE + 1, BLOCK_SIZE):
            blocks.append(Ipad[x:x+BLOCK_SIZE, y:y+BLOCK_SIZE].reshape(-1))
    return np.stack(blocks, axis=0)


def col2im(X_col: np.ndarray, row, col, row_new, col_new) -> np.ndarray:
    """Reconstruct image from block columns, crop to original size."""
    X_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new - BLOCK_SIZE + 1, BLOCK_SIZE):
        for y in range(0, col_new - BLOCK_SIZE + 1, BLOCK_SIZE):
            X_rec[x:x+BLOCK_SIZE, y:y+BLOCK_SIZE] = X_col[count].reshape(BLOCK_SIZE, BLOCK_SIZE)
            count += 1
    return X_rec[:row, :col]


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return ssim(img1, img2, data_range=255)
