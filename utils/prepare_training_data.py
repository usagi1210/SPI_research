"""
从 BSD400 提取 33x33 patch，保存为 Training_Data.mat。

可以从任意目录运行：
    python utils/prepare_training_data.py
    python prepare_training_data.py
"""
import os
import argparse
import numpy as np
import scipy.io as sio
import cv2
from glob import glob

PATCH_SIZE = 33


def extract_patches(img: np.ndarray, patch_size: int, stride: int) -> list:
    h, w = img.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(img[y:y+patch_size, x:x+patch_size].reshape(-1))
    return patches


def main():
    parser = argparse.ArgumentParser(description='Prepare BSD400 training patches')
    _here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--img_dir',    type=str,
                        default=os.path.join(_here, '../data/train/BSD400'),
                        help='BSD400 图片目录')
    parser.add_argument('--out_dir',    type=str,
                        default=os.path.join(_here, '../data/train'),
                        help='输出目录，生成 Training_Data.mat')
    parser.add_argument('--patch_size', type=int, default=PATCH_SIZE)
    parser.add_argument('--stride',     type=int, default=14,
                        help='patch 提取步长，越小 patch 越多（默认 14，约 20 万 patch）')
    args = parser.parse_args()

    img_paths = sorted(
        glob(os.path.join(args.img_dir, '*.jpg')) +
        glob(os.path.join(args.img_dir, '*.png')) +
        glob(os.path.join(args.img_dir, '*.bmp'))
    )

    if not img_paths:
        print(f'[ERROR] 未找到图片：{args.img_dir}')
        return

    all_patches = []
    for path in img_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'  [SKIP] 无法读取：{path}')
            continue
        img = img.astype(np.float32) / 255.0
        all_patches.extend(extract_patches(img, args.patch_size, args.stride))

    labels = np.array(all_patches, dtype=np.float32)   # (N, 1089)
    print(f'共提取 {len(labels)} 个 patch，来自 {len(img_paths)} 张图')

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'Training_Data.mat')
    sio.savemat(out_path, {'labels': labels})
    print(f'已保存：{out_path}')


if __name__ == '__main__':
    main()
