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

# ── Threshold settings ───────────────────────────────────────
#
# FIXED_PASS_THRESHOLD — Layer 1 of the threshold locking system.
#
#   Set this to a float to lock the PASS threshold permanently.
#   It will NEVER change no matter how many times you retrain.
#
#   HOW TO USE:
#     After your first training run, the terminal prints:
#       "Computed value : 0.7483  (74.8%)"
#     Copy that number here:
#       FIXED_PASS_THRESHOLD = 0.748
#
#   Leave it as None to let the lock-file system manage it automatically
#   (threshold is computed once on first training run, then never changes).
#
#   None  = use lock file system (recommended)
#   float = always use this exact value (maximum control)
FIXED_PASS_THRESHOLD = None   # e.g. set to 0.748 to hardcode 74.8%

# Three-zone thresholds used as fallback when threshold.pkl is missing.
# These are overridden by FIXED_PASS_THRESHOLD or the lock file.
PASS_THRESHOLD  = 0.65   # fallback PASS zone bottom
# WARN_THRESHOLD  = 0.45   # fallback WARNING zone bottom

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