# Vision Transformers for Irregular Time Series: Depression Prediction with StudentLife Data

Master's thesis project — Vrije Universiteit Amsterdam (2025)

This repository contains the code for a feasibility study on predicting depressive symptom trajectories from multimodal smartphone sensing data, using the **Vision Transformer for Irregularly Sampled Time Series (ViTST)** architecture applied to the [StudentLife dataset](https://studentlife.cs.dartmouth.edu/).

---

## What this project does

The StudentLife dataset captures 10 weeks of passive smartphone sensor data (GPS, audio, activity, app usage) and ecological momentary assessments (EMA: mood, stress, sleep) from 48 Dartmouth undergraduate students. The goal was to predict depressive symptom deterioration — defined as a ≥5-point increase on the PHQ-9 scale — from behavioural time-series data alone.

The core idea behind ViTST is to encode multivariate time series as images, feeding them into a Vision Transformer (ViT) that performs self-attention across both the temporal and feature axes — without requiring imputation or temporal regularisation. Alongside the sensor images, static survey features are encoded with a parallel BERT encoder and fused before classification.

This work extended the original ViTST pipeline (Li et al., NeurIPS 2023) to the StudentLife dataset, which required:

- Building a full data preprocessing pipeline for 10 sensor modalities and EMA streams
- Defining and engineering 12 daily-resolution behavioural features across 4 modalities
- Implementing binary classification support with clinically motivated label definitions (PHQ-9 deterioration and depression-status)
- Adding evaluation metrics suited to imbalanced settings (AUROC, AUPRC, F1)
- Experimenting with lightweight synthetic data augmentation to address extreme class imbalance

---

## Key finding

The pipeline is technically sound — validated against the original ViTST benchmarks — but the StudentLife dataset proved too sparse and imbalanced for reliable positive-class detection (4–9 depressed cases out of 48 participants). The model defaulted to majority-class predictions across all leave-one-user-out folds, achieving near-zero recall. The conclusion is that the data constraints, not the architecture, are the primary bottleneck.

---

## Results summary

| Experiment | Accuracy | AUROC | AUPRC | F1 |
|---|---|---|---|---|
| Random baseline | 54.3 ± 49.8% | 0.0 | 0.0 | 2.9% |
| Deterioration (standard) | 91.4 ± 28.0% | — | 8.6 ± 28.0% | 0.0% |
| Deterioration (+ synthetic aug.) | 72.6 ± 41.8% | 3.2 ± 8.9% | 24.1 ± 38.2% | 0.9% |
| Depression-status (PHQ-9 threshold) | 74.3 ± 43.7% | — | 25.7 ± 43.7% | 0.0% |

High accuracy here is misleading — it reflects majority-class collapse. AUPRC and AUROC are the meaningful metrics given severe class imbalance.

---

## Repository structure

```
├── run_VisionTextCLS.py        # Main training and evaluation script (this repo)
├── models/
│   └── vision_text_dual_encoder/   # ViTST dual-encoder architecture
├── load_data.py                # Data loading and split utilities
├── splits/                     # Leave-one-user-out split files
├── dataset/                    # Not included — download from https://studentlife.cs.dartmouth.edu/dataset.html
├── vitst-env.yml               # Conda environment
└── README.md
```

The preprocessing pipeline (sensor aggregation, feature extraction, image generation) runs separately before this script and produces the dataset consumed here.

---

## Dataset

The StudentLife dataset is not included in this repository. Download it from [https://studentlife.cs.dartmouth.edu/dataset.html](https://studentlife.cs.dartmouth.edu/dataset.html) and follow the preprocessing steps to generate the required files (`Pdict_list.npy`, `arr_outcomes.npy`, split files, and time series images). Place them under `dataset/studentlife_images/` following the structure expected by `load_data.py`.

## Setup

```bash
conda env create -f vitst-env.yml
conda activate vitst-env
```

**Important:** Update the `sys.path.insert` line near the top of `run_VisionTextCLS.py`, `run_VisionTextCLS1_with_random.py`, and `models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py` to point to your local code directory. For example:
```python
sys.path.insert(0, "/path/to/your/repo/code")
```
This is standard practice for this codebase — the original ViTST repo requires the same step.

---

## Running experiments

```bash
# Depression-status classification with ViT + BERT
python run_VisionTextCLS.py \
  --dataset StudentLife \
  --image_model vit \
  --text_model bert \
  --do_train \
  --epochs 50 \
  --train_batch_size 8 \
  --learning_rate 2e-5 \
  --n_splits 5
```

---

## Based on

- **ViTST**: Li, Z., Li, S., & Yan, X. (2023). *Time series as images: Vision transformer for irregularly sampled time series.* NeurIPS 2023. [[Paper]](https://arxiv.org/abs/2303.12799) · [[Original Repo]](https://github.com/Leezekun/ViTST)
- **StudentLife dataset**: Wang et al. (2014). *StudentLife: Assessing mental health, academic performance and behavioral trends of college students using smartphones.* UbiComp 2014. [[Dataset]](https://studentlife.cs.dartmouth.edu/dataset.html) · [[Paper]](https://dl.acm.org/doi/10.1145/2632048.2632054)

The core ViTST architecture and training framework were adapted from the original authors' codebase. The StudentLife data pipeline, label definitions, binary classification support, and evaluation extensions were developed as part of this thesis.

---

## Requirements

Python 3.9, PyTorch 2.1.0 (CUDA 11.8), HuggingFace Transformers 4.28.1. Full dependency list in `vitst-env.yml`.
