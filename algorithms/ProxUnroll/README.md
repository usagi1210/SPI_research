# Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging

**CVPR 2025** — [[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Proximal_Algorithm_Unrolling_Flexible_and_Efficient_Reconstruction_Networks_for_Single-Pixel_CVPR_2025_paper.html)] [[arXiv](https://arxiv.org/abs/2505.23180)]

[Ping Wang](https://scholar.google.com/citations?user=WCsIUToAAAAJ&hl=zh-CN&oi=ao), [Lishun Wang](https://scholar.google.com/citations?user=BzkbrCgAAAAJ&hl=zh-CN&oi=sra), [Gang Qu](https://scholar.google.com/citations?user=AvPlPSUAAAAJ&hl=zh-CN&oi=sra), [Xiaodong Wang](https://scholar.google.com/citations?user=2JXMfrcAAAAJ&hl=zh-CN&oi=sra), [Yulun Zhang](https://scholar.google.com/citations?user=ORmLjWoAAAAJ&hl=zh-CN), [Xin Yuan](https://scholar.google.com/citations?user=cS9CbWkAAAAJ&hl=zh-CN)

## Abstract

Deep-unrolling and plug-and-play (PnP) approaches have become the de-facto standard solvers for single-pixel imaging (SPI) inverse problem. PnP approaches, a class of iterative algorithms where regularization is implicitly performed by an off-the-shelf deep denoiser, are flexible for varying compression ratios (CRs) but are limited in reconstruction accuracy and speed. Conversely, unrolling approaches, a class of multi-stage neural networks where a truncated iterative optimization process is transformed into an end-to-end trainable network, typically achieve better accuracy with faster inference but require fine-tuning or even retraining when CR changes. In this paper, we address the challenge of integrating the strengths of both classes of solvers. To this end, we design an efficient deep image restorer (DIR) for the unrolling of HQS (half quadratic splitting) and ADMM (alternating direction method of multipliers). More importantly, a general proximal trajectory (PT) loss function is proposed to train HQS/ADMM-unrolling networks such that learned DIR approximates the proximal operator of an ideal explicit restoration regularizer. Extensive experiments demonstrate that the resulting proximal unrolling networks can not only flexibly handle varying CRs with a single model like PnP algorithms, but also outperform previous CR-specific unrolling networks in both reconstruction accuracy and speed.

<div align="center">
  <img src="https://github.com/pwangcs/ProxUnroll/blob/main/fig/summary.png" width="800">
  <br>
  <b>TL;DR:</b> ProxUnroll achieves SOTA performance with high flexibility and fast convergence.
</div>

## ProxUnroll

<div align="center">
  <img src="https://github.com/pwangcs/ProxUnroll/blob/main/fig/proxunroll.png" width="800">
  <br>
  Proximal algorithm unrolling via trajectory loss.
</div>

<div align="center">
  <img src="https://github.com/pwangcs/ProxUnroll/blob/main/fig/network.png" width="800">
  <br>
  Deep image restorer \(\mathcal{R}_{\theta}\) used in ProxUnroll.
</div>

## Result

<div align="center">
  <img src="https://github.com/pwangcs/ProxUnroll/blob/main/fig/result.png" width="800">
</div>
<div align="center">
  <img src="https://github.com/pwangcs/ProxUnroll/blob/main/fig/simulated_visualization.png" width="800">
</div>
<div align="center">
  <img src="https://github.com/pwangcs/ProxUnroll/blob/main/fig/real_visualization.png" width="800">
</div>

---

## Getting Started

### Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- Dependencies: `numpy`, `scipy`, `opencv-python`, `einops`, `timm`, `albumentations`, `scikit-image`, `scikit-learn`

Example installation:

```bash
pip install torch torchvision numpy scipy opencv-python einops timm albumentations scikit-image scikit-learn
```

Run all commands from the **repository root** so that relative paths (e.g. `measurement_matrix/`) resolve correctly.

### Repository layout

```
ProxUnroll/
├── model/
│   └── proxunroll.py      # Unified HQS / ADMM unrolling network
├── measurement_matrix/    # Learned sensing matrices (.mat), required at runtime
├── weight/                # Pretrained checkpoints (download separately)
├── opts.py                # Shared CLI arguments
├── train_proxunroll.py    # Training
├── test_proxunroll.py     # Evaluation
├── utils.py               # Datasets, metrics, logging, checkpoints
├── fig/                   # Figures for this README
└── ../../results/ProxUnroll/  # Standardized outputs (created automatically)
```

### Solver selection (`--solver`)

Both **HQS** and **ADMM** unrolling are implemented in a single model and controlled by one flag:

| `--solver` | Description | Default checkpoint name |
|------------|-------------|-------------------------|
| `hqs`      | Half-Quadratic Splitting (HQS) Unrolling (default) | `weight/hqs_proxunroll.pth` |
| `admm`     | Alternating Direction Method of Multipliers (ADMM) Unrolling | `weight/admm_proxunroll.pth` |

The flag also sets internal run names (`hqs_proxunroll` / `admm_proxunroll`) used for log and checkpoint folders.

---

## Data preparation

### Training

- **BSDS400** (or any folder of RGB training images).
- Set path with `--train_data_path`.
- Each iteration samples random crops and resizes them to one or more fixed resolutions (Y channel in YCrCb), controlled by `--train_sizes`:
  - `256_321` — **256×256** and **321×481** only (lower GPU memory; recommended if 512×512 does not fit).
  - `256_321_512` — **256×256**, **321×481**, and **512×512** (default; full multi-scale training).

### Testing

| Split | Role | CLI argument | Typical use |
|-------|------|--------------|-------------|
| **Set11** | Grayscale (Y channel) | `--test_data_path` | `--solver hqs` / `admm`, gray eval |
| **CBSD68** | Color (Y reconstructed, Cr/Cb from GT) | `--test_color_data_path` | Color eval |

Place images in folders of plain `.png` / `.jpg` files. Evaluation uses CR ∈ `{0.01, 0.04, 0.10, 0.25, 0.50}`.

### Measurement matrices

Ensure these files exist under `measurement_matrix/`:

- `blind_learned_256_256_matrices.mat`
- `blind_learned_321_481_matrices.mat`
- `blind_learned_512_512_matrices.mat`

Training and inference only support resolutions **256×256**, **321×481**, and **512×512**.

---

## Training

Train with proximal trajectory (PT) loss. Compression ratio is randomized over  
`[0.01, 0.04, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]` during optimization.

**HQS (default):**

```bash
python train_proxunroll.py \
  --solver hqs \
  --train_data_path /path/to/BSDS400 \
  --train_sizes 256_321 \
  --epochs 200 \
  --batch_size 1 \
  --lr 1e-3 --lr_min 1e-4
```

Use `--train_sizes 256_321_512` to include 512×512 patches (higher memory).

**ADMM:**

```bash
python train_proxunroll.py \
  --solver admm \
  --train_data_path /path/to/BSDS400 \
  --epochs 200
```

**Resume from checkpoint:**

```bash
python train_proxunroll.py \
  --solver hqs \
  --train_data_path /path/to/BSDS400 \
  --pretrained_model_path ../../results/ProxUnroll/ProxUnroll-HQS-crmulti--<timestamp>/checkpoints/epoch_10.pth
```

**Run validation during training** (Set11 + CBSD68 at fixed CRs; slower):

```bash
python train_proxunroll.py \
  --solver hqs \
  --train_data_path /path/to/BSDS400 \
  --test_flag True \
  --test_data_path /path/to/Set11 \
  --test_color_data_path /path/to/CBSD68
```

**Multi-GPU (DDP):**

```bash
torchrun --nproc_per_node=4 train_proxunroll.py \
  --solver hqs \
  --distributed True \
  --train_data_path /path/to/BSDS400
```

### Standardized outputs

Training uses multiple compression ratios, so each run is stored as:

```text
../../results/ProxUnroll/ProxUnroll-HQS-crmulti--YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── epoch_*.pth
│   ├── latest_crmulti.pth
│   └── best_cr25.pth               # Written when --test_flag true
├── logs/
│   └── train.log
├── vis/
│   ├── train/
│   └── test/epoch_*/{gray,color}/cr*/
└── best_per_image_metrics.csv      # Best gray Set11 result at --primary_cr
```

Standalone evaluation creates one directory per sampling ratio, for example
`ProxUnroll-HQS-cr25--YYYYMMDD_HHMMSS/`, containing `checkpoints/`, `logs/`,
`vis/`, and `per_image_metrics_gray.csv` / `per_image_metrics_color.csv`.

Use `--result_dir`, `--run_id`, and `--primary_cr` to customize the root,
run identifier, and validation criterion respectively.
For a specific GPU, pass `--device cuda:1`; the training script honors this
device in non-distributed mode.

### Fixed-CR training with epoch-wise validation

To match fixed-compression-ratio SPI experiments, pass `--train_cr` and enable
validation. For example, this trains and evaluates a 25% model on Set11 at the
end of every epoch:

```bash
python train_proxunroll.py \
  --solver hqs \
  --train_cr 0.25 \
  --eval_crs 0.25 \
  --primary_cr 0.25 \
  --test_flag true \
  --test_every 1
```

The log prints `Epoch | Set11 | CR | PSNR | SSIM` at every validation epoch.
The run is named `ProxUnroll-HQS-cr25--YYYYMMDD_HHMMSS`, and the best model is
stored as `checkpoints/best_cr25.pth`. Omit `--train_cr` to retain the original
multi-CR training schedule. Use `--eval_color true` only when CBSD68 validation
is also needed.

---

## Testing

Evaluate a trained checkpoint on Set11 (gray) and CBSD68 (color):

```bash
python test_proxunroll.py \
  --solver hqs \
  --test_model_path ./weight/hqs_proxunroll.pth \
  --test_data_path /path/to/Set11 \
  --test_color_data_path /path/to/CBSD68
```

```bash
python test_proxunroll.py \
  --solver admm \
  --test_model_path ./weight/admm_proxunroll.pth \
  --test_data_path /path/to/Set11 \
  --test_color_data_path /path/to/CBSD68
```

If `--test_model_path` is omitted, the script defaults to `./weight/{solver}_proxunroll.pth`.

### Test outputs

- Reconstructions: `../../results/ProxUnroll/ProxUnroll-{HQS,ADMM}-crXX--<timestamp>/vis/{gray,color}/`
- Logs: the corresponding `logs/test_gray.log` and `logs/test_color.log`
- Per-image metrics: `per_image_metrics_gray.csv` and `per_image_metrics_color.csv`

---

## Common arguments

All scripts share options from `opts.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--solver` | `hqs` | `hqs` or `admm` |
| `--train_sizes` | `256_321_512` | `256_321` or `256_321_512` training resolutions |
| `--epochs` | `200` | Training epochs |
| `--lr` | `1e-3` | Peak learning rate (cosine schedule start) |
| `--lr_min` | `1e-4` | Minimum learning rate (cosine schedule end) |
| `--batch_size` | `1` | Batch size |
| `--dim` | `48` | Restorer base channel width |
| `--enc_blocks` | `[2,2,2]` | Encoder block counts per stage |
| `--dec_blocks` | `[2,2,2]` | Decoder block counts per stage |
| `--mid_blocks` | `2` | Bottleneck blocks |
| `--iter_step` | `2000` | Log every N iterations |
| `--save_train_image_step` | `2000` | Save reconstruction grid every N iterations |
| `--save_model_step` | `1` | Save checkpoint every N epochs |
| `--pretrained_model_path` | `None` | Resume training |
| `--test_model_path` | `None` | Checkpoint for `test_proxunroll.py` |
| `--device` | `cuda` | Device for single-GPU runs |
| `--result_dir` | `../../results/ProxUnroll` | Standardized results root |
| `--run_id` | `None` | Optional timestamp / experiment identifier |
| `--primary_cr` | `0.25` | Gray validation CR used to select the best checkpoint |
| `--train_cr` | `None` | Fixed training CR; omit for original multi-CR training |
| `--eval_crs` | train CR / standard CRs | CRs evaluated during training |
| `--test_every` | `1` | Validation interval in epochs |
| `--eval_color` | `false` | Include CBSD68 while training |
| `--test_crs` | `1,4,10,25,50%` | CRs evaluated by the standalone test script |
| `--test_color` | `false` | Include CBSD68 in standalone testing |
| `--torchcompile` | `None` | Optional `torch.compile` backend (e.g. `inductor`) |

---

## Pretrained models

Place released weights under `weight/`:

The checkpoint files are intentionally excluded from this repository. Transfer
them to the target machine separately when needed.

```
weight/
├── hqs_proxunroll.pth
└── admm_proxunroll.pth
```

Then run `test_proxunroll.py` with the matching `--solver`.

---

## Citation

If you use ProxUnroll, please cite:

```bibtex
@inproceedings{wang2025proxunroll,
  title={Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging},
  author={Wang, Ping and Wang, Lishun and Qu, Gang and Wang, Xiaodong and Zhang, Yulun and Yuan, Xin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={411--421},
  year={2025}
}
```

## Contact

Questions: [wangping@westlake.edu.cn](mailto:wangping@westlake.edu.cn)
