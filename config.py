# ============================================================
# config.py  —  All project settings in one place
# ============================================================

# ── Audio settings ──────────────────────────────────────────
SAMPLE_RATE   = 22050   # Audio samples per second
DURATION      = 10      # Max seconds to read from each file
N_MFCC        = 40      # Number of MFCC coefficients (RF model)
N_MELS        = 128     # Number of mel filterbanks
HOP_LENGTH    = 512     # Hop size for STFT
N_FFT         = 2048    # FFT window size

# ── Folder paths ─────────────────────────────────────────────
DATASET_DIR   = "dataset"
NORMAL_DIR    = "dataset/normal"      # Put GOOD engine audio here
ABNORMAL_DIR  = "dataset/abnormal"   # Put FAULTY engine audio here
MODEL_DIR     = "models"
MODEL_PATH    = "models/engine_model.pkl"
SCALER_PATH   = "models/scaler.pkl"

# ── Baseline settings  (used by acoustic_baseline.py) ────────
BASELINE_N_MFCC     = 20
BASELINE_PATH       = "models/baseline.pkl"
BASELINE_CSV        = "models/baseline_features.csv"
BASELINE_PLOT       = "models/avg_spectrogram.png"
BASELINE_WARN_SIGMA = 2.0    # yellow warning
BASELINE_FAIL_SIGMA = 3.5    # red alert

# ── Training settings ────────────────────────────────────────
TEST_SIZE     = 0.2     # 20 % data used for testing
RANDOM_STATE  = 42      # Reproducibility seed
N_ESTIMATORS  = 200     # Number of trees in Random Forest

# =============================================================================
# SOLUTION SETTINGS  (all lines below are NEW — added to fix false positives)
# =============================================================================

# ── Task 7: Three-zone Gray Zone decision thresholds ─────────
# Instead of binary PASS/FAIL we now use three zones:
#   PASS    — Normal Score >= PASS_THRESHOLD  (engine is healthy)
#   WARNING — Normal Score >= WARN_THRESHOLD  (borderline, re-test advised)
#   FAIL    — Normal Score <  WARN_THRESHOLD  (engine has a problem)
#
# IMPORTANT: run_training() auto-calculates an adaptive PASS_THRESHOLD
# from the actual training data and saves it to models/threshold.pkl.
# That file takes priority over this hardcoded value.
# These values are only used as a fallback when threshold.pkl is missing.
PASS_THRESHOLD  = 0.65   # CHANGED — was 0.90 (too strict for small datasets)
WARN_THRESHOLD  = 0.45   # CHANGED — new: bottom of WARNING zone

# ── Task 4: Frequency Band Filtering ─────────────────────────
# Engine fundamental frequencies are concentrated in 500–2000 Hz.
# Applying a bandpass filter before feature extraction removes:
#   - Low-frequency rumble (HVAC, traffic below 500 Hz)
#   - High-frequency hiss (microphone noise above 2000 Hz)
# Both of these are common causes of false positives.
BANDPASS_LOW    = 500    # CHANGED — new: Hz lower bound
BANDPASS_HIGH   = 2000   # CHANGED — new: Hz upper bound

# ── Task 5: Smoothing window ─────────────────────────────────
# A moving-average smoothing is applied to the mel-spectrogram features
# before they are passed to the model.  This removes short spikes that
# can mislead the classifier into thinking the engine is abnormal.
SMOOTHING_WIN   = 5      # CHANGED — new: frames (odd number works best)

# ── Task 3 & 9: Feature group weights ────────────────────────
# Not all features are equally reliable.  MFCCs capture the core
# timbral texture (most diagnostic), RMS and spectral features are
# medium-reliability, and chroma/ZCR are often noisy.
# These weights are used in the weighted z-score computation.
WEIGHT_MFCC     = 0.50   # CHANGED — new: 50% weight on MFCC features
WEIGHT_SPECTRAL = 0.30   # CHANGED — new: 30% weight on RMS + spectral
WEIGHT_OTHER    = 0.20   # CHANGED — new: 20% weight on chroma + ZCR

# ── Task 6 & 10: Ensemble voting ─────────────────────────────
# Final FAIL verdict requires at least ENSEMBLE_FAIL_VOTES out of
# 3 independent detectors to agree:
#   Detector 1 — Random Forest probability (RF model)
#   Detector 2 — Baseline sigma distance   (acoustic_baseline.py)
#   Detector 3 — Per-feature z-score rule  (predict.py)
#
# Setting this to 2 means a single uncertain detector cannot
# alone cause a false positive — at least one other must confirm.
ENSEMBLE_FAIL_VOTES = 2  # CHANGED — new: votes needed to declare FAIL