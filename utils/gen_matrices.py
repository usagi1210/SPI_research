"""Generate Gaussian random measurement matrices for LoRA-DUN (64×64 patches).

Output: matrices/phi_{cr}_{N}.mat  with variable 'phi', shape (M, N)
Each row is normalised to unit L2 norm for numerical stability.

Usage (run from project root):
    python utils/gen_matrices.py
    python utils/gen_matrices.py --save_dir matrices --seed 42
"""
import os
import argparse
import numpy as np
import scipy.io as sio

SAMPLING_RATES = [0.01, 0.04, 0.10, 0.25, 0.40, 0.50]
PATCH_SIZE     = 64          # 64×64 patches
N              = PATCH_SIZE ** 2   # 4096


def gen_matrix(cr: float, N: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M   = max(1, int(round(cr * N)))
    Phi = rng.standard_normal((M, N)).astype(np.float32)
    # Row-normalise: each measurement vector has unit L2 norm
    norms = np.linalg.norm(Phi, axis=1, keepdims=True)
    Phi   = Phi / norms
    return Phi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='matrices')
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--rates',    type=float, nargs='+',
                        default=SAMPLING_RATES)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    for cr in args.rates:
        Phi      = gen_matrix(cr, N, args.seed)
        filename = f'phi_{cr}_{N}.mat'
        out_path = os.path.join(args.save_dir, filename)
        sio.savemat(out_path, {'phi': Phi})
        print(f'Saved {filename}  shape={Phi.shape}  M={Phi.shape[0]}')

    print(f'\nAll matrices saved to: {os.path.abspath(args.save_dir)}')


if __name__ == '__main__':
    main()
