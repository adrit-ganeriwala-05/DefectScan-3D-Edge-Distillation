<div align="center">

# DefectScan AI

### Edge-Optimised 3D Printing Defect Detection  
### via Knowledge Distillation and Dynamic INT8 Quantisation

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.63%25-brightgreen)]()
[![FPS](https://img.shields.io/badge/Edge%20FPS-152.4-orange)]()

*MobileNetV4-Conv-Medium Teacher → MobileNetV4-Conv-Small Student → Dynamic INT8*

</div>

---

## Overview

DefectScan AI is a real-time quality control system for FDM 3D printers that detects three critical defect classes directly on edge hardware — no cloud, no GPU required at inference.

The pipeline uses **Knowledge Distillation** (KL Divergence + Attention Transfer) to compress a high-accuracy teacher model into a student that is **3.4× smaller**, **1.93× faster**, and runs at **152 FPS on 4 CPU threads** with zero accuracy loss after Dynamic INT8 quantisation.

| Metric | Value |
|---|---|
| Test Accuracy (FP32 = INT8) | **99.63%** |
| Macro F1 | **0.9922** |
| Edge FPS (4-thread CPU, no CUDA) | **152.4** |
| Avg Latency | **6.56 ms** |
| Jitter σ | **0.44 ms** |
| Student INT8 Size | **2.91 MB** |
| Speedup vs Teacher | **1.93×** |

---

## Defect Classes

| Class | Description | Severity |
|---|---|---|
| 🔴 **Spaghetti** | Filament collapsed mid-print, tangled strands | Critical — stop print |
| 🔶 **Stringing** | Fine threads between features from excess extrusion | Moderate |
| 🔴 **Zits** | Surface blobs from over-extrusion | Minor |

---

## Architecture

```
Caxton Dataset (180k images)  ──► FAILED (label noise, bad variance)
                                          │
                                          ▼
Roboflow Dataset (8,851 images) ──► MobileNetV4-Conv-Medium (Teacher, 8.7M params)
                                          │  Knowledge Distillation
                                          │  CE + KL Divergence + Attention Transfer
                                          ▼
                                   MobileNetV4-Conv-Small (Student FP32, 2.5M params)
                                          │  Dynamic INT8 Quantisation
                                          ▼
                                   Student INT8 (2.91 MB) ──► Edge Deployment
```

---

## Results

### Ablation Study

| Loss Configuration | Val Acc | Test Acc | Macro F1 |
|---|---|---|---|
| CE only | 97.91% | 97.61% | 0.9541 |
| CE + KL | 99.18% | 99.04% | 0.9822 |
| **CE + KL + AT (full)** | **99.71%** | **99.63%** | **0.9922** |

### Edge Benchmark (4-thread CPU, CUDA disabled)

| Metric | Teacher FP32 | Student INT8 |
|---|---|---|
| Avg Latency | 12.68 ms | **6.56 ms** |
| P95 Latency | 15.47 ms | **7.41 ms** |
| Jitter σ | 1.32 ms | **0.44 ms** |
| FPS | 78.84 | **152.38** |
| Speedup | — | **1.93×** |

---

## Repository Structure

```
DefectScan-3D-Edge-Distillation/
│
├── src/
│   ├── phase1_data_pipeline.py        # Hardware check, augmentations, WeightedRandomSampler
│   ├── phase1_train_teacher.py        # MobileNetV4-Medium teacher training
│   ├── phase2b_visual_eval.py         # Teacher visual sanity check
│   ├── phase2c_test_teacher_ood.py    # Teacher OOD single-image inference
│   ├── phase3a_train_distill.py       # Staged KL + AT Knowledge Distillation
│   ├── phase3b_eval_testset.py        # FP32 student test set evaluation
│   ├── phase3c_eval_ood.py            # FP32 student OOD inference
│   ├── phase4a_quantize_int8.py       # Dynamic INT8 quantisation
│   ├── phase4b_eval_testset.py        # INT8 evaluation + edge benchmark
│   ├── phase4c_eval_ood.py            # INT8 OOD inference
│   ├── ood_visual_test.py             # Side-by-side 3-model OOD comparison
│   └── demo_app.py                    # Streamlit MLOps dashboard
│
├── outputs/
│   ├── figures/                       # All training curves, confusion matrices, benchmarks
│   └── models/                        # Saved model weights (not tracked by git)
│
├── dataset/                           # Not tracked — see Dataset Setup below
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/adrit-ganeriwala-05/DefectScan-3D-Edge-Distillation.git
cd DefectScan-3D-Edge-Distillation
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` installs CPU-only PyTorch by default.  
> For GPU training (RTX 4070 Super), install the CUDA 12.1 build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Dataset Setup

This project uses the **3D Printing Failure** dataset from Roboflow Universe.

1. Go to [https://universe.roboflow.com/3d-printing-failure/3d-printing-failure](https://universe.roboflow.com/3d-printing-failure/3d-printing-failure)
2. Export as **PyTorch (ImageFolder)** format
3. Place the downloaded folder as `dataset/` in the project root:

```
dataset/
├── train/
│   ├── spaghetti/
│   ├── stringing/
│   └── zits/
├── val/
│   └── ...
└── test/
    └── ...
```

---

## Pipeline Execution

Run scripts from the **project root** with the virtual environment active.

```bash
# Phase 1 — Verify hardware and data pipeline
python src/phase1_data_pipeline.py

# Phase 2 — Train the teacher
python src/phase1_train_teacher.py

# Phase 3 — Knowledge Distillation
python src/phase3a_train_distill.py

# Phase 3b — Evaluate FP32 student
python src/phase3b_eval_testset.py

# Phase 4a — Quantise to INT8
python src/phase4a_quantize_int8.py

# Phase 4b — Evaluate INT8 + edge benchmark
python src/phase4b_eval_testset.py

# OOD visual test (all 3 models side by side)
python src/ood_visual_test.py https://your-image-url.jpg
# or with a local file:
python src/ood_visual_test.py path/to/image.jpg
```

---

## Streamlit Dashboard

```bash
streamlit run src/demo_app.py
```

Opens at `http://localhost:8501`. Features:
- Upload image or use live webcam
- Confidence bars per defect class with severity badge
- Dark / Light mode toggle
- Performance metrics strip (accuracy, FPS, model size, speedup)

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW (fused) |
| Learning rate | 3 × 10⁻⁵ |
| Weight decay | 10⁻² |
| Batch size | 64 |
| Epochs | 150 |
| Early stopping | Macro F1, patience = 25 |
| LR schedule | CosineAnnealingWarmRestarts (T₀=30, T_mult=2) |
| KD temperature T | 5.0 |
| α / β / γ | 0.4 / 0.4 / 0.2 |
| Label smoothing | 0.05 |

**Distillation loss:**

$$\mathcal{L} = \alpha \mathcal{L}_{CE} + \beta \mathcal{L}_{KL} + \gamma \mathcal{L}_{AT}$$

---

## Hardware

Trained on:
- **GPU:** NVIDIA GeForce RTX 4070 Super (12 GB VRAM)
- **OS:** Windows 10
- **Python:** 3.12
- **PyTorch:** 2.5.1+cu121

Edge simulation benchmarks ran on 4-thread CPU with `CUDA_VISIBLE_DEVICES="-1"`.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{ganeriwala2025defectscan,
  author    = {Adrit Ganeriwala},
  title     = {DefectScan AI: Edge-Optimised 3D Printing Defect Detection
               via Knowledge Distillation and Dynamic INT8 Quantisation},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/adrit-ganeriwala-05/DefectScan-3D-Edge-Distillation}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with PyTorch 2.5 · timm · Streamlit · Georgia State University</sub>
</div>