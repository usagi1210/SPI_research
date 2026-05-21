# ISTA-Net+

Reproduction of **ISTA-Net+** (CVPR 2018).

> Jian Zhang and Bernard Ghanem, "ISTA-Net: Interpretable Optimization-Inspired Deep Network
> for Image Compressive Sensing", CVPR 2018.

## Data & Matrices

Put files in the shared project directories (paths are relative to this folder):

| What | Where |
|------|-------|
| Training patches | `../../data/train/Training_Data.mat` |
| Set11 test images | `../../data/test/Set11/*.tif` |
| BSD68 test images | `../../data/test/BSD68/*.png` |
| Sampling matrices | `../../matrices/phi_0_{ratio}_1089.mat` |

**Download links (original repo):**
- Training data & matrices: https://drive.google.com/open?id=1AoEcNA5-onnSqBcWZawNw7ZFrJ1fFR_C
- PyTorch repo: https://github.com/jianzhangcs/ISTA-Net-PyTorch

## Train

```bash
# ratio=25%, 9 phases, default settings
python train.py --cs_ratio 25 --num_layers 9 --epochs 200

# resume from epoch 100
python train.py --cs_ratio 25 --num_layers 9 --resume_epoch 100
```

## Test

```bash
# test on Set11, load epoch-200 checkpoint
python test.py --cs_ratio 25 --num_layers 9 --epoch_num 200 --test_set Set11

# test on BSD68
python test.py --cs_ratio 25 --num_layers 9 --epoch_num 200 --test_set BSD68
```

## Reproduce all ratios

```bash
for ratio in 1 4 10 25 40 50; do
    python train.py --cs_ratio $ratio
    python test.py  --cs_ratio $ratio --epoch_num 200
done
```

## Results location

Checkpoints, logs, and reconstructed images are saved under `../../results/ISTA_Net/`.
