# SPI Research

Single-Pixel Imaging (SPI) algorithm research project.

## Structure

```text
SPI_research/
├── algorithms/
│   ├── ISTA_Net/    ISTA-Net+ (CVPR 2018) — comparison baseline
│   └── ours/        Proposed algorithm: LoRA-DUN
├── data/
│   ├── train/       BSD400 raw images  (not tracked by git)
│   └── test/
│       ├── Set11/   11 standard test images
│       └── BSD68/   68 BSD test images
├── matrices/        Sampling matrices (not tracked by git)
├── results/         Experiment outputs (not tracked by git)
├── paper/           LaTeX manuscript
└── utils/           Shared utility functions
```

## Datasets

| Split | Dataset | Note |
|-------|---------|------|
| Train | BSD400 | 400 natural images, random 64×64 crop at runtime |
| Test  | Set11  | Standard CS benchmark |
| Test  | BSD68  | Berkeley Segmentation Dataset |

## Algorithms

| Method    | Paper     | Status      |
|-----------|-----------|-------------|
| ISTA-Net+ | CVPR 2018 | In progress |
| LoRA-DUN  | Ours      | In progress |

## Workflow

1. Edit code locally
2. Push to GitHub
3. `git pull` on the server and run experiments
