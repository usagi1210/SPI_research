# SPI Research

Single-Pixel Imaging (SPI) algorithm research project.

## Structure

```text
SPI_research/
├── algorithms/
│   ├── ISTA_Net/    ISTA-Net+ (CVPR 2018) — comparison baseline
│   └── proposed/    Proposed algorithm (TBD)
├── data/
│   ├── train/       Training_Data.mat  (not tracked by git)
│   └── test/
│       ├── Set11/   11 standard test images
│       └── BSD68/   68 BSD test images
├── matrices/        Sampling matrices, phi_0_{ratio}_1089.mat
├── results/         Experiment outputs (not tracked by git)
├── paper/           LaTeX manuscript
└── utils/           Shared utility functions
```

## Datasets

| Split | Dataset                                    | Source                 |
|-------|--------------------------------------------|------------------------|
| Train | Training_Data.mat (91-image patches, 88912 blocks) | ISTA-Net Google Drive |
| Test  | Set11                                      | Standard CS benchmark  |
| Test  | BSD68                                      | Berkeley Segmentation  |

## Algorithms

| Method     | Paper     | Status      |
|------------|-----------|-------------|
| ISTA-Net+  | CVPR 2018 | In progress |
| Proposed   | —         | TBD         |

## Workflow

1. Edit code locally
2. Push to GitHub via Claude Code
3. `git pull` on the server and run experiments
