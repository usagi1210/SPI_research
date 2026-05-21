# CLAUDE.md — SPI Research Project

## 项目概述

单像素成像（Single-Pixel Imaging, SPI）算法科研项目，目标是提出一个新的 SPI 重建算法，发表论文。

研究者有 CASSI（编码孔径快照光谱成像）的科研背景，熟悉仿真+真实实验的完整论文流程。

---

## 工作流

1. **本地**（Windows, `f:\Research\Project\SPI`）修改代码
2. **Claude** 负责 `git commit` + `git push` 到 GitHub
3. **服务器** 执行 `git pull` 后运行实验

GitHub 仓库：`https://github.com/usagi1210/SPI_research`（Public）

---

## 目录结构

```text
SPI_research/
├── algorithms/
│   ├── ISTA_Net/       对比算法 1：ISTA-Net+（CVPR 2018）
│   └── proposed/       待开发：本文提出的算法
├── data/
│   ├── train/          Training_Data.mat（不上传 git）
│   └── test/
│       ├── Set11/      主测试集（11张经典图，.tif）
│       └── BSD68/      辅测试集（68张图）
├── matrices/           采样矩阵 phi_0_{ratio}_1089.mat（不上传 git）
├── results/            实验输出，不上传 git
├── paper/              LaTeX 论文
│   ├── figures/
│   └── tables/
└── utils/              各算法共用工具函数（预留）
```

**git 策略**：`data/`、`results/`、`*.mat`、`*.npy` 等大文件不上传，只上传代码和目录骨架（`.gitkeep`）。

---

## 数据集

| 用途 | 数据集 | 说明 |
|------|--------|------|
| 训练 | BSD400 / Training_Data.mat | 88912 个 33×33 patch，原始来自 91 张自然图 |
| 测试（主） | Set11 | CS 领域黄金标准，所有对比算法统一用这个 |
| 测试（辅） | BSD68 | 样本更多，统计更稳 |
| 采样矩阵 | phi_0_{ratio}_1089.mat | 高斯随机矩阵，ratio ∈ {1,4,10,25,30,40,50} |

下载地址（ISTA-Net Google Drive）：`https://drive.google.com/open?id=1AoEcNA5-onnSqBcWZawNw7ZFrJ1fFR_C`

---

## 实验规划

### 仿真实验（必做）
- [ ] 标准重建基准：Set11 + BSD68，报告 PSNR / SSIM
- [ ] 多采样率：1%, 4%, 10%, 25%, 40%, 50%
- [ ] 噪声鲁棒性：测量值加高斯噪声，不同 SNR
- [ ] 消融实验：验证各模块有效性
- [ ] 计算复杂度：推理时间 + 参数量对比

### 真实实验（一区必做）
- [ ] 静态场景重建（DMD + 单光电探测器）
- [ ] 不同采样率下的真实重建
- [ ] 特殊场景（低光照 / 动态 / 散射介质，至少一组）

---

## 对比算法列表

| 方法 | 论文 | 代码位置 | 状态 |
|------|------|----------|------|
| ISTA-Net+ | CVPR 2018 | `algorithms/ISTA_Net/` | **代码完成，待跑实验** |
| （待定） | — | — | 未开始 |

---

## 当前进度

### 已完成
- [x] GitHub 仓库初始化，工作流配置完毕
- [x] 项目目录结构建立
- [x] ISTA-Net+ PyTorch 复现代码编写完成
  - `model.py`：`BasicBlock` + `ISTANetPlus`
  - `train.py`：训练循环，路径指向共享 `data/` 和 `matrices/`
  - `test.py`：测试，输出 PSNR/SSIM，保存重建图
  - `utils.py`：`imread_cs` / `img2col` / `col2im` / `compute_psnr` / `compute_ssim`

### 下一步
1. 服务器上下载数据（Training_Data.mat、采样矩阵、Set11）
2. 跑 ISTA-Net+ 训练（`python train.py --cs_ratio 25`）
3. 跑 ISTA-Net+ 测试，验证复现结果与论文一致
4. 开发 `proposed` 算法

---

## 技术栈

- Python + PyTorch
- 图像质量指标：PSNR、SSIM（skimage）
- 数据格式：`.mat`（scipy.io）、`.tif`/`.png`（opencv）
- 服务器：Linux，`git pull` 同步代码

---

## 给 Claude 的工作规范

- 修改代码后主动提示是否需要 `git commit` + `git push`
- 每次推送前确认 commit message 清晰描述改动
- 新增对比算法时，在 `algorithms/` 下新建独立子目录，结构参考 `ISTA_Net/`
- 大文件（数据、模型权重）不加入 git，路径用相对路径 `../../data/` 指向共享目录
- 更新此文件：每当有重要进度变化时同步更新"当前进度"部分
