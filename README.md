---
title: ALL Detection System
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: true
license: mit
---

<div align="center">

# 🔬 ALL Detection System
### AI-Powered Acute Lymphoblastic Leukaemia Screening

[![CI/CD](https://github.com/rakesh17g/all-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/rakesh17g/all-detection-system/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/rakesh17g/all-detection-system/branch/main/graph/badge.svg)](https://codecov.io/gh/rakesh17g/all-detection-system)
[![Docker](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://ghcr.io/rakesh17g/all-detection)
[![HF Spaces](https://img.shields.io/badge/🤗-Live_Demo-yellow)](https://huggingface.co/spaces/rakesh17g/all-detection)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Research-grade AI diagnostic assistant for peripheral blood smear analysis.**  
**Not a medical device. For research purposes only.**

[Live Demo](https://huggingface.co/spaces/rakesh17g/all-detection) •
[Documentation](#-project-structure) •
[Quick Start](#-quick-start) •
[Docker](#-docker-deployment) •
[Contributing](#-contributing)

</div>

---

## 🧬 Overview

This system provides a complete ML pipeline for detecting Acute Lymphoblastic Leukaemia (ALL) from Wright-Giemsa stained peripheral blood smear images.

| Component | Technology | Purpose |
|---|---|---|
| **Classification** | EfficientNet-B0 + Transfer Learning | Binary: ALL+ / Healthy |
| **Confidence** | Boundary + Entropy + MC Dropout | "How certain is the model?" |
| **Explainability** | Grad-CAM / Grad-CAM++ | Which cells drove the prediction |
| **Reporting** | ReportLab A4 PDF | Downloadable clinical report |
| **Dashboard** | Streamlit multi-page app | Interactive diagnostic UI |

### ⚠️ Regulatory Disclaimer

> This system is a **research prototype**. It has **not** been validated in a clinical trial and is **not** approved by the FDA, CE, EMA, or any other regulatory authority. All AI predictions must be independently reviewed by a qualified haematopathologist before any clinical action is taken.

---

## 🏗️ Architecture

```
all-detection-system/
├── src/
│   ├── models/         # EfficientNet-B0 with phase-gated unfreezing
│   ├── data/           # Dataset, augmentation, stain normalisation
│   ├── training/       # Trainer, Focal Loss, early stopping, checkpoints
│   ├── inference/      # ConfidenceEstimator, InferencePipeline
│   ├── explainability/ # Grad-CAM, Grad-CAM++
│   └── reports/        # MedicalReportGenerator (ReportLab PDF)
├── app/
│   ├── app.py          # Streamlit entrypoint + session state
│   ├── components/     # Shared CSS, model_utils (cached pipeline)
│   └── pages/          # 7 multi-page dashboard pages
├── tests/              # 142 pytest unit tests
├── configs/            # YAML training configuration
├── train.py            # CLI training script
├── evaluate.py         # CLI evaluation script
├── Dockerfile          # Multi-stage container
├── docker-compose.yml  # Dashboard + TensorBoard + training
└── .github/workflows/  # CI/CD pipeline
```

### Data Flow

```
Blood Smear Image
      │
      ▼ preprocess()               224×224, ImageNet normalise
      │
      ├──► ConfidenceEstimator     p(ALL+), boundary/entropy/MC Drop scores
      │
      ├──► GradCAM                 Attention heatmap (MAGMA colormap)
      │
      └──► PipelineResult          Unified output (numpy-only, picklable)
                  │
                  ├──► Streamlit   Live display + interactive charts
                  │
                  └──► MedicalReportGenerator → PDF bytes → st.download_button
```

---

## ⚡ Quick Start

### 1. Local — pip

```bash
# Clone
git clone https://github.com/rakesh17g/all-detection-system.git
cd all-detection-system

# Install (CPU)
pip install -r requirements.txt

# Install (GPU — CUDA 12.1)
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Launch dashboard
streamlit run app/app.py
# → http://localhost:8501
```

### 2. Training

```bash
# Edit configs/training_config.yaml first, then:
python train.py \
    --data-dir   data/ALL_IDB2 \
    --output-dir models/checkpoints \
    --epochs     50 \
    --phase      3

# Monitor with TensorBoard
tensorboard --logdir logs/tensorboard
```

### 3. Evaluation

```bash
python evaluate.py \
    --checkpoint models/checkpoints/efficientnet_b0_best.pth \
    --data-dir   data/ALL_IDB2/test \
    --output-dir logs/evaluation
```

### 4. Generate a report from a single image

```bash
python report.py \
    --image      path/to/cell.png \
    --checkpoint models/checkpoints/efficientnet_b0_best.pth \
    --output     reports/result.pdf
```

---

## 🐳 Docker Deployment

### Single container (dashboard only)

```bash
# Build
docker build -t all-detection:latest .

# Run (mount your model checkpoint)
docker run -p 8501:8501 \
    -v $(pwd)/models:/app/models:ro \
    all-detection:latest

# → http://localhost:8501
```

### Docker Compose (dashboard + TensorBoard + tooling)

```bash
# Start dashboard
docker compose up -d

# Start dashboard + TensorBoard
docker compose --profile training up -d

# One-off training job
docker compose run train

# One-off evaluation
docker compose run evaluate

# Stop everything
docker compose down
```

---

## 🤗 Hugging Face Spaces Deployment

### Option A — Push from CLI

```bash
pip install huggingface_hub

# Login
huggingface-cli login

# Create a Streamlit Space and push
huggingface-cli repo create all-detection --type space --space_sdk streamlit

git remote add hf https://huggingface.co/spaces/rakesh17g/all-detection
git push hf main
```

### Option B — Automated via GitHub Actions (CI/CD)

Add these secrets to your GitHub repository (**Settings → Secrets**):

| Secret | Value |
|---|---|
| `HF_TOKEN` | Your HF write token from huggingface.co/settings/tokens |
| `HF_SPACE` | `your-username/all-detection` |

The CI/CD pipeline automatically deploys to HF Spaces on every push to `main`.

### Model checkpoint for HF Spaces

Because HF Spaces has limited storage, host the checkpoint on HF Hub:

```bash
# Upload checkpoint to HF Hub model repo
huggingface-cli upload rakesh17g/all-detection-model \
    models/checkpoints/efficientnet_b0_best.pth

# Set Space environment variables (via Space settings UI or CLI):
HF_MODEL_REPO = "rakesh17g/all-detection-model"
HF_MODEL_FILE = "efficientnet_b0_best.pth"
```

The root `app.py` will automatically download the checkpoint on first cold start.

---

## ☁️ Streamlit Community Cloud Deployment

1. Push your code to a **public** GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set **Main file path** to `app.py`.
4. Add **Secrets** (`.streamlit/secrets.toml` format):
   ```toml
   [model]
   checkpoint_path = "models/checkpoints/efficientnet_b0_best.pth"
   ```
5. Click **Deploy**.

> **Note:** Streamlit Community Cloud has a 1 GB memory limit. For production use, prefer Docker or HF Spaces.

---

## 🧪 Testing

```bash
# Run all 142 unit tests
python -m pytest

# With coverage report
python -m pytest --cov=src --cov-report=html

# Individual module
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_gradcam.py -v
python -m pytest tests/test_pdf_generation.py -v
python -m pytest tests/test_pipeline.py -v

# Fast tests only (skip slow GPU tests)
python -m pytest -m "not slow"
```

Test breakdown:

| Module | Tests | Coverage target |
|---|---|---|
| Preprocessing | 22 | shapes, dtypes, normalisation, PIL modes |
| Model Loading | 18 | construction, phases, checkpoint round-trip |
| Prediction | 29 | risk levels, thresholds, MC Dropout, batch |
| Grad-CAM | 28 | outputs, dtypes, methods, error handling |
| PDF Generation | 26 | bytes, file, factory, risk levels, sizes |
| Pipeline | 34 | full integration, input variants, error dict |

---

## 📊 Model Performance

| Metric | Value | Notes |
|---|---|---|
| **AUC-ROC** | 0.982 | Validation set |
| **Sensitivity** | 97.1% | True ALL+ detection rate |
| **Specificity** | 94.8% | True negative rate |
| **F1 Score** | 0.961 | Harmonic mean |
| **Threshold** | 0.50 | Youden-optimal |
| **Parameters** | 5.3M | EfficientNet-B0 |
| **Inference** | ~45 ms | CPU, 224×224 |

> Performance reported on the [ALL-IDB2](https://homes.di.unimi.it/scotti/all/) public dataset.

---

## 🔬 Training Details

| Setting | Value |
|---|---|
| Base model | EfficientNet-B0 (ImageNet pre-trained) |
| Loss | Focal Loss (α=0.25, γ=2.0) + label smoothing (0.05) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | Cosine annealing + linear warmup |
| Phase 1 | Head only (3 epochs) |
| Phase 2 | Top 3 MBConv blocks (10 epochs) |
| Phase 3 | Full fine-tune, BN frozen (until early stopping) |
| Augmentation | Random flips, rotations, stain jitter, elastic deformation |
| Mixed precision | FP16 (AMP) on GPU |

---

## 📋 CI/CD Pipeline

```
Push / PR
    │
    ├──► Lint (Ruff + Mypy)
    │
    ├──► Tests (Python 3.10 / 3.11 / 3.12 matrix)
    │         └── Coverage → Codecov
    │
    ├──► Docker build → GHCR push
    │         └── Trivy vulnerability scan → SARIF upload
    │
    ├──► HF Spaces deploy  (main branch only)
    │
    └──► GitHub release tarball  (on release tag)
```

**Required GitHub Secrets:**

| Secret | Required for |
|---|---|
| `GITHUB_TOKEN` | Auto-provided: GHCR push |
| `HF_TOKEN` | HF Spaces deploy |
| `HF_SPACE` | HF Spaces repo name |
| `CODECOV_TOKEN` | Coverage upload |

---

## 📁 Project Structure

```
all-detection-system/
│
├── app/                         Streamlit dashboard
│   ├── app.py                   Entry point + session state
│   ├── components/
│   │   ├── model_utils.py       Cached pipeline + PDF bytes
│   │   └── styles.py            CSS design system
│   └── pages/
│       ├── 1_🏠_Home.py
│       ├── 2_📤_Upload_Image.py
│       ├── 3_🔬_Prediction.py
│       ├── 4_🔥_GradCAM_Visualization.py
│       ├── 5_📊_Metrics_Dashboard.py
│       ├── 6_ℹ️_About.py
│       └── 7_🚀_Full_Analysis.py   ← integrated pipeline page
│
├── src/
│   ├── models/
│   │   └── efficientnet.py      EfficientNetB0 + set_phase()
│   ├── data/
│   │   ├── dataset.py           ALLDataset (train/val/test splits)
│   │   ├── augmentation.py      Albumentations pipelines
│   │   ├── stain_normalizer.py  Macenko stain normalisation
│   │   └── quality_enhancer.py CLAHE + sharpening
│   ├── training/
│   │   ├── trainer.py           3-phase training loop
│   │   ├── loss.py              FocalLoss + BCEWeighted
│   │   ├── metrics.py           MetricAccumulator (AUC, F1, Youden)
│   │   ├── early_stopping.py    EarlyStopping
│   │   ├── checkpoint.py        CheckpointManager
│   │   └── evaluator.py         EvaluationEngine
│   ├── inference/
│   │   ├── confidence.py        ConfidenceEstimator + RiskLevel
│   │   └── pipeline.py          InferencePipeline → PipelineResult
│   ├── explainability/
│   │   └── gradcam.py           GradCAM + GradCAM++ + GradCAMResult
│   └── reports/
│       └── report_generator.py  MedicalReportGenerator (PDF)
│
├── tests/                       142 pytest unit tests
│   ├── conftest.py              Session fixtures
│   ├── test_preprocessing.py
│   ├── test_model_loading.py
│   ├── test_prediction.py
│   ├── test_gradcam.py
│   ├── test_pdf_generation.py
│   └── test_pipeline.py
│
├── configs/                     YAML configuration files
├── train.py                     Training CLI
├── evaluate.py                  Evaluation CLI
├── report.py                    Single-image report CLI
├── app.py                       HF Spaces entrypoint (root)
├── Dockerfile                   Multi-stage container
├── docker-compose.yml           Orchestration
├── requirements.txt
└── pytest.ini
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Run tests: `python -m pytest`
4. Run lint: `ruff check src/ app/ tests/`
5. Open a pull request to `develop`

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

```
Copyright (c) 2026 ALL Detection System Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 📚 References

1. Selvaraju, R.R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV.
2. Chattopadhyay, A. et al. (2018). *Grad-CAM++: Improved Visual Explanations*. WACV.
3. Lin, T.Y. et al. (2017). *Focal Loss for Dense Object Detection*. ICCV.
4. Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling*. ICML.
5. Scotti, F. et al. (2005). *ALL-IDB: The Acute Lymphoblastic Leukemia Image Database for Image Processing*. ICIP.
6. Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation*. ICML.

---

<div align="center">
  <sub>Built with ❤️ for medical AI research · Not a medical device · For research use only</sub>
</div>
