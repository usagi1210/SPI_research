"""
生成重建效果对比图（原图 vs 各算法重建），保存到 paper/figures/。

用法：
    # 单张图，多算法对比
    python utils/visualize.py --image barbara --cs_ratio 25

    # 多张图，指定算法
    python utils/visualize.py --image barbara boats cameraman --cs_ratio 25 --algorithms ISTA_Net

    # 单算法多采样率对比
    python utils/visualize.py --image barbara --cs_ratio 4 10 25 50 --algorithms ISTA_Net --mode ratio
"""
import os
import re
import glob
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.normpath(os.path.join(_HERE, '..'))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
DATA_DIR    = os.path.join(ROOT_DIR, 'data', 'test')
PAPER_DIR   = os.path.join(ROOT_DIR, 'paper', 'figures')


def find_original(test_set, img_name):
    """在 data/test/{test_set}/ 下找匹配图片。"""
    folder = os.path.join(DATA_DIR, test_set)
    for ext in ('*.tif', '*.png', '*.bmp', '*.jpg'):
        matches = glob.glob(os.path.join(folder, ext))
        for p in matches:
            if img_name.lower() in os.path.basename(p).lower():
                return p
    return None


def find_reconstructed(algo, test_set, cs_ratio, img_name):
    """在 results/{algo}/images/{test_set}/ratio_{cs_ratio}/ 下找匹配图片。"""
    folder = os.path.join(RESULTS_DIR, algo, 'images', test_set, f'ratio_{cs_ratio}')
    if not os.path.isdir(folder):
        return None, None, None
    for p in glob.glob(os.path.join(folder, '*.png')):
        base = os.path.basename(p).lower()
        if img_name.lower() in base:
            # 从文件名解析 PSNR / SSIM
            m = re.search(r'PSNR([0-9.]+)_SSIM([0-9.]+)', p)
            psnr = float(m.group(1)) if m else None
            ssim = float(m.group(2)) if m else None
            return p, psnr, ssim
    return None, None, None


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0 if img is not None else None


def make_comparison_figure(image_names, algorithms, cs_ratio, test_set, save_path):
    """
    行 = 图片，列 = [原图] + [各算法]
    """
    n_rows = len(image_names)
    n_cols = 1 + len(algorithms)
    fig_w  = n_cols * 2.5
    fig_h  = n_rows * 2.5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    col_labels = ['Original'] + algorithms
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].set_title(label, fontsize=9, fontweight='bold')

    for row_idx, img_name in enumerate(image_names):
        # 原图
        orig_path = find_original(test_set, img_name)
        orig = load_gray(orig_path) if orig_path else None
        ax = axes[row_idx, 0]
        if orig is not None:
            ax.imshow(orig, cmap='gray', vmin=0, vmax=1)
            ax.set_ylabel(img_name, fontsize=8, rotation=0, labelpad=40, va='center')
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

        # 各算法重建
        for col_idx, algo in enumerate(algorithms, start=1):
            rec_path, psnr, ssim = find_reconstructed(algo, test_set, cs_ratio, img_name)
            ax = axes[row_idx, col_idx]
            if rec_path:
                rec = load_gray(rec_path)
                ax.imshow(rec, cmap='gray', vmin=0, vmax=1)
                if psnr is not None:
                    ax.set_xlabel(f'PSNR={psnr:.2f}\nSSIM={ssim:.4f}', fontsize=7)
            else:
                ax.text(0.5, 0.5, 'Not found', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    plt.suptitle(f'{test_set}  |  CS ratio={cs_ratio}%', fontsize=10, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'已保存：{save_path}')


def make_ratio_figure(img_name, algorithm, cs_ratios, test_set, save_path):
    """
    列 = [原图] + [各采样率重建]，单张图展示不同采样率效果
    """
    n_cols = 1 + len(cs_ratios)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.5, 3))

    orig_path = find_original(test_set, img_name)
    orig = load_gray(orig_path) if orig_path else None
    axes[0].imshow(orig if orig is not None else np.zeros((64, 64)), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original', fontsize=9, fontweight='bold')
    axes[0].axis('off')

    for i, ratio in enumerate(cs_ratios, start=1):
        rec_path, psnr, ssim = find_reconstructed(algorithm, test_set, ratio, img_name)
        ax = axes[i]
        if rec_path:
            rec = load_gray(rec_path)
            ax.imshow(rec, cmap='gray', vmin=0, vmax=1)
            label = f'CS {ratio}%'
            if psnr is not None:
                label += f'\n{psnr:.2f} dB'
            ax.set_title(label, fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    plt.suptitle(f'{algorithm}  |  {img_name}  |  {test_set}', fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'已保存：{save_path}')


def main():
    parser = argparse.ArgumentParser(description='生成重建对比图')
    parser.add_argument('--image',      type=str, nargs='+', default=['barbara'],
                        help='要可视化的图片名（不含扩展名）')
    parser.add_argument('--algorithms', type=str, nargs='+', default=None,
                        help='算法名（results/ 下的子目录名），默认自动检测所有')
    parser.add_argument('--cs_ratio',   type=int, nargs='+', default=[25])
    parser.add_argument('--test_set',   type=str, default='Set11')
    parser.add_argument('--mode',       type=str, default='algo',
                        choices=['algo', 'ratio'],
                        help='algo: 多算法对比同一采样率; ratio: 单算法多采样率对比')
    args = parser.parse_args()

    # 自动检测 results/ 下有结果的算法
    if args.algorithms is None:
        if os.path.exists(RESULTS_DIR):
            args.algorithms = [
                d for d in sorted(os.listdir(RESULTS_DIR))
                if os.path.isdir(os.path.join(RESULTS_DIR, d, 'images'))
            ]
        else:
            args.algorithms = []

    if not args.algorithms:
        print('[ERROR] 未找到任何算法结果，请先运行 test.py。')
        return

    if args.mode == 'algo':
        for ratio in args.cs_ratio:
            save_path = os.path.join(
                PAPER_DIR,
                f'comparison_{args.test_set}_ratio{ratio}.png'
            )
            make_comparison_figure(args.image, args.algorithms, ratio, args.test_set, save_path)

    elif args.mode == 'ratio':
        algo = args.algorithms[0]
        for img_name in args.image:
            save_path = os.path.join(
                PAPER_DIR,
                f'ratios_{algo}_{img_name}_{args.test_set}.png'
            )
            make_ratio_figure(img_name, algo, args.cs_ratio, args.test_set, save_path)


if __name__ == '__main__':
    main()
