---
title: Engine Acoustic Diagnostic AI
emoji: 🔊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.31.0
app_file: app.py
pinned: false
license: mit
---

# 🔊 Engine Acoustic Diagnostic AI

An AI-powered system that detects engine faults from audio recordings.

## Features

- **Engine Diagnosis** — Upload a WAV/MP3 recording → get PASS / WARNING / FAIL
- **Acoustic Baseline** — Build a healthy-engine fingerprint from multiple recordings
- **Multi-File Analysis** — Batch statistics, FFT, spectrograms, anomaly detection
- **Worm Graph** — Cricket-style cumulative energy comparison
- **Ensemble Voting** — 3 independent detectors reduce false positives

## How to Use

### Tab 1 — Engine Diagnosis
1. Upload a single engine audio file
2. Click **Analyze Engine**
3. See PASS / WARNING / FAIL result with confidence score, baseline comparison, worm graph, and anomaly indicators

### Tab 2 — Baseline Builder
1. Upload healthy engine WAV files OR use files already in `dataset/normal/`
2. Click **Build Baseline from Uploaded Files**
3. Baseline fingerprint is saved and used automatically in Tab 1

### Tab 3 — Multi-File Analysis
1. Upload multiple WAV files or enter a folder path
2. Click **Analyse All Files**
3. See statistics, FFT, spectrograms, group comparison, and anomaly detection

## Technical Stack

- **Audio analysis**: Librosa
- **Machine learning**: Scikit-learn (Random Forest + ensemble)
- **UI**: Gradio
- **Signal processing**: SciPy

## Model Setup

Upload your pre-trained models to the `models/` folder:
- `models/engine_model.pkl` — trained Random Forest
- `models/scaler.pkl` — feature scaler
- `models/baseline.pkl` — acoustic baseline (built in Tab 2)
- `models/threshold.pkl` — adaptive threshold (auto-generated during training)
- `models/feature_importances.pkl` — feature weights (auto-generated during training)