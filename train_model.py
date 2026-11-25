# ============================================================
# train_model.py  —  Train the AI model and save it to disk
#
# THRESHOLD LOCKING (latest update):
#
#   Problem:  The adaptive threshold (PASS %) was recomputed from
#             scratch every training run, so it changed whenever
#             the dataset size changed:
#               - 21 samples  → 74.8%
#               - 88 samples  → 85.0%
#             This caused confusion because the same engine recording
#             could PASS one day and FAIL the next, just because more
#             training data was added.
#
#   Root cause in the old code:
#             compute_adaptive_threshold() used np.percentile(normal_probs, 5)
#             which depends on BOTH the number of samples AND the model's
#             confidence level — both of which change with every retrain.
#
#   Fix:      Three-layer threshold protection:
#
#     Layer 1 — FIXED_PASS_THRESHOLD in config.py
#               A hardcoded value you set once and own.
#               Set it after your first good training run.
#               Use:  FIXED_PASS_THRESHOLD = 0.748  (from your 1st run)
#               When this is set (not None), it ALWAYS wins.
#
#     Layer 2 — LOCK FILE  (models/threshold_locked.pkl)
#               Written once when you first run training.
#               On subsequent retrains, the locked value is reloaded
#               instead of recomputed.
#               Delete this file ONLY when you deliberately want a new
#               threshold (e.g. after adding a completely new engine type).
#
#     Layer 3 — Adaptive fallback
#               If no lock file and no fixed config value, the old
#               percentile method runs as before.  This only triggers
#               on the very first training run.
#
#   How to use:
#     First run   — threshold is computed and lock file is created
#     Retrains    — lock file is found, same threshold is reused
#     Reset       — delete models/threshold_locked.pkl, retrain once
#     Hard-fix    — set FIXED_PASS_THRESHOLD = 0.748 in config.py
# ============================================================

import os
import numpy as np
import joblib
from sklearn.ensemble         import RandomForestClassifier
from sklearn.svm              import SVC
from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics          import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)

from data_preparation import prepare_dataset, create_project_folders
from config import (
    MODEL_PATH, SCALER_PATH, TEST_SIZE,
    RANDOM_STATE, N_ESTIMATORS,
    MODEL_DIR,
    FIXED_PASS_THRESHOLD,     # LOCK — use this to hardcode the threshold in config.py
)

# Paths for saved artefacts
THRESHOLD_PATH      = "models/threshold.pkl"
THRESHOLD_LOCK_PATH = "models/threshold_locked.pkl"   # LOCK — written once, reused on retrains
IMPORTANCES_PATH    = "models/feature_importances.pkl"


# ────────────────────────────────────────────────────────────
# 1.  FEATURE SCALING
# ────────────────────────────────────────────────────────────
def scale_features(X_train, X_test):
    """Normalise features to mean=0, std=1."""
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ────────────────────────────────────────────────────────────
# 2.  MODEL BUILDERS
# ────────────────────────────────────────────────────────────
def build_random_forest():
    """Random Forest — robust on small datasets."""
    model = RandomForestClassifier(
        n_estimators      = N_ESTIMATORS,
        max_depth         = 15,
        min_samples_split = 4,
        class_weight      = "balanced",
        random_state      = RANDOM_STATE,
        n_jobs            = -1
    )
    return model


def build_svm():
    """SVM alternative."""
    model = SVC(
        kernel       = "rbf",
        C            = 10,
        gamma        = "scale",
        probability  = True,
        class_weight = "balanced",
        random_state = RANDOM_STATE
    )
    return model


# ────────────────────────────────────────────────────────────
# 3.  TRAIN
# ────────────────────────────────────────────────────────────
def train_model(X_train, y_train, algorithm="random_forest"):
    """Train the chosen model."""
    if algorithm == "svm":
        model = build_svm()
        print("Training Support Vector Machine ...")
    else:
        model = build_random_forest()
        print("Training Random Forest ...")
    model.fit(X_train, y_train)
    print("   Training complete!")
    return model


# ────────────────────────────────────────────────────────────
# 4.  EVALUATE
# ────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    """Print accuracy, confusion matrix, full classification report."""
    y_pred   = model.predict(X_test)
    y_prob   = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc      = roc_auc_score(y_test, y_prob)

    print("\n" + "="*55)
    print("  MODEL EVALUATION RESULTS")
    print("="*55)
    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print(f"  AUC-ROC  : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Normal  (correct)   : {cm[0][0]}")
    print(f"  False Alarm  (FP)        : {cm[0][1]}")
    print(f"  Missed Fault (FN)        : {cm[1][0]}")
    print(f"  True Fault   (correct)   : {cm[1][1]}")
    print("="*55)
    return accuracy


def cross_validate_model(model, X, y, n_splits=5):
    """K-Fold cross validation."""
    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"\n  {n_splits}-Fold Cross Validation:")
    print(f"  Mean: {scores.mean()*100:.2f}%  |  Std: {scores.std()*100:.2f}%")
    return scores


# ────────────────────────────────────────────────────────────
# 5.  SAVE AND LOAD MODEL
# ────────────────────────────────────────────────────────────
def save_model(model, scaler):
    """Save trained model and scaler."""
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  Model saved  -> {MODEL_PATH}")
    print(f"  Scaler saved -> {SCALER_PATH}")


def load_model():
    """Load model + scaler from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}.  Run: python train_model.py")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}.  Run: python train_model.py")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# ────────────────────────────────────────────────────────────
# 6.  THRESHOLD MANAGEMENT  (Three-layer locking system)
# ────────────────────────────────────────────────────────────

def _compute_threshold_from_data(model, X_train_scaled, y_train):
    """
    Internal helper — compute the 5th-percentile threshold from training data.
    This runs ONLY when no lock file and no fixed config value exists.
    After the first run, the result is locked and this never runs again.

    Returns: (threshold, mean_prob, std_prob)
    """
    probs        = model.predict_proba(X_train_scaled)
    normal_mask  = (y_train == 0)
    normal_probs = probs[normal_mask, 0]

    if len(normal_probs) == 0:
        return 0.65, 0.65, 0.0

    mean_prob = float(np.mean(normal_probs))
    std_prob  = float(np.std(normal_probs))
    p5        = float(np.percentile(normal_probs, 5))
    threshold = float(np.clip(p5, 0.40, 0.85))

    print(f"\n  Adaptive threshold computed from {int(normal_mask.sum())} normal training samples:")
    print(f"    Mean prob_normal  : {mean_prob:.4f}")
    print(f"    Std  prob_normal  : {std_prob:.4f}")
    print(f"    5th percentile    : {p5:.4f}")
    print(f"    Computed value    : {threshold:.4f}  ({threshold*100:.1f}%)")
    return threshold, mean_prob, std_prob


def resolve_threshold(model, X_train_scaled, y_train):
    """
    Three-layer threshold resolution.  Checks layers in order and
    stops at the first layer that provides a value.

    LAYER 1 — config.py hardcoded value (FIXED_PASS_THRESHOLD)
    ───────────────────────────────────────────────────────────
    If FIXED_PASS_THRESHOLD is set to a float in config.py, it wins
    unconditionally.  No file is read, no computation is done.
    Use this when you want 100% control:
        FIXED_PASS_THRESHOLD = 0.748   # in config.py

    LAYER 2 — lock file (models/threshold_locked.pkl)
    ──────────────────────────────────────────────────
    Written after the FIRST training run.  Every subsequent retrain
    reads this file and reuses the same value.
    To reset: delete models/threshold_locked.pkl and retrain once.

    LAYER 3 — compute from data (first run only)
    ─────────────────────────────────────────────
    Runs only when neither Layer 1 nor Layer 2 applies.
    Computes the 5th-percentile of normal-class probabilities.
    Immediately writes the lock file so Layer 2 takes over next time.

    Returns: (threshold, mean_prob, std_prob, source_label)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Layer 1: fixed config value ───────────────────────────
    if FIXED_PASS_THRESHOLD is not None:
        threshold = float(FIXED_PASS_THRESHOLD)
        print(f"\n  Threshold — Layer 1 (config.py fixed value):")
        print(f"    FIXED_PASS_THRESHOLD = {threshold:.4f}  ({threshold*100:.1f}%)")
        print(f"    This value is always used regardless of training data.")
        return threshold, threshold, 0.0, "config_fixed"

    # ── Layer 2: lock file ────────────────────────────────────
    if os.path.exists(THRESHOLD_LOCK_PATH):
        locked = joblib.load(THRESHOLD_LOCK_PATH)
        threshold = float(locked["threshold"])
        print(f"\n  Threshold — Layer 2 (lock file reused from first training run):")
        print(f"    Locked value      : {threshold:.4f}  ({threshold*100:.1f}%)")
        print(f"    Locked on         : {locked.get('locked_on', 'unknown date')}")
        print(f"    Lock file         : {THRESHOLD_LOCK_PATH}")
        print(f"    To reset          : delete {THRESHOLD_LOCK_PATH} and retrain")
        return threshold, locked.get("mean_prob", threshold), locked.get("std_prob", 0.0), "lock_file"

    # ── Layer 3: compute from data (first run) ────────────────
    print(f"\n  Threshold — Layer 3 (first run — computing from training data):")
    print(f"    No lock file found at {THRESHOLD_LOCK_PATH}")
    print(f"    Computing threshold and writing lock file ...")
    threshold, mean_prob, std_prob = _compute_threshold_from_data(
        model, X_train_scaled, y_train
    )
    # Write the lock file immediately so every future retrain reuses this value
    from datetime import datetime
    lock_data = {
        "threshold"  : threshold,
        "mean_prob"  : mean_prob,
        "std_prob"   : std_prob,
        "locked_on"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "locked_from": "first_training_run",
    }
    joblib.dump(lock_data, THRESHOLD_LOCK_PATH)
    print(f"    Lock file written : {THRESHOLD_LOCK_PATH}")
    print(f"    This value will be reused on ALL future retrains.")
    print(f"    To reset          : delete {THRESHOLD_LOCK_PATH} and retrain")
    return threshold, mean_prob, std_prob, "computed_and_locked"


def save_threshold(threshold, mean_prob, std_prob, source_label):
    """
    Save the FINAL resolved threshold to models/threshold.pkl.
    This is what predict.py reads at inference time.
    The value written here is always the one from resolve_threshold()
    — so it is always stable after the first training run.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    data = {
        "threshold" : threshold,
        "mean_prob" : mean_prob,
        "std_prob"  : std_prob,
        "source"    : source_label,
    }
    joblib.dump(data, THRESHOLD_PATH)
    print(f"  Threshold saved   : {THRESHOLD_PATH}  "
          f"(value: {threshold:.4f} = {threshold*100:.1f}%,  source: {source_label})")


def reset_threshold_lock():
    """
    Utility — call this when you WANT the threshold to be recomputed
    on the next training run (e.g. after adding a new engine type or
    changing recording conditions).

    Usage:
        from train_model import reset_threshold_lock
        reset_threshold_lock()
        run_training()   # threshold will be recomputed and a new lock written
    """
    if os.path.exists(THRESHOLD_LOCK_PATH):
        os.remove(THRESHOLD_LOCK_PATH)
        print(f"  Lock file deleted: {THRESHOLD_LOCK_PATH}")
        print(f"  Next training run will recompute the threshold from data.")
    else:
        print(f"  No lock file found at {THRESHOLD_LOCK_PATH} — nothing to reset.")


# ────────────────────────────────────────────────────────────
# 7.  FEATURE IMPORTANCES  (NEW — Task 3)
# ────────────────────────────────────────────────────────────

def save_feature_importances(model):
    """
    CHANGED (Task 3) — Save RF feature_importances_ to disk.

    feature_importances_ is an array of shape (n_features,) where
    each value is the mean decrease in impurity contributed by that
    feature.  Higher = more diagnostic for distinguishing normal/abnormal.

    predict.py loads this and uses it to weight the per-feature z-score
    computation: noisy low-importance features are down-weighted so they
    cannot alone trigger a false positive.
    """
    if not hasattr(model, "feature_importances_"):
        print("  Note: model has no feature_importances_ (SVM?), skipping save.")
        return

    importances = model.feature_importances_         # shape (n_features,)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(importances, IMPORTANCES_PATH)
    top5_idx  = np.argsort(importances)[::-1][:5]
    print(f"  Feature importances saved -> {IMPORTANCES_PATH}")
    print(f"  Top-5 most important feature indices: {list(top5_idx)}")


# ────────────────────────────────────────────────────────────
# 8.  MAIN TRAINING PIPELINE
# ────────────────────────────────────────────────────────────
def run_training(algorithm="random_forest"):
    """
    Full end-to-end training pipeline.

    CHANGED: now also computes adaptive threshold (Task 1) and saves
    feature importances (Task 3) after training.
    """
    create_project_folders()

    # Step 1 — Load data
    X, y = prepare_dataset()
    if len(X) < 10:
        print("\n  WARNING: Very small dataset.  Add more audio files for better accuracy.")

    # Step 2 — Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y
    )
    print(f"\n  Data split — Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Step 3 — Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 4 — Train
    print()
    model = train_model(X_train_scaled, y_train, algorithm=algorithm)

    # Step 5 — Evaluate
    evaluate_model(model, X_test_scaled, y_test)

    # Cross-validate if enough samples
    if np.sum(y == 0) >= 10 and np.sum(y == 1) >= 10:
        print("\n  Running cross-validation ...")
        X_all_scaled = scaler.transform(X)
        cross_validate_model(model, X_all_scaled, y)

    # Step 6 — Save model + scaler
    save_model(model, scaler)

    # LOCK Step 7 — Resolve threshold using 3-layer locking system
    # Layer 1: config.py FIXED_PASS_THRESHOLD  (if set, always wins)
    # Layer 2: models/threshold_locked.pkl      (reuse from first run)
    # Layer 3: compute from data                (first run only, then locks)
    threshold, mean_prob, std_prob, source = resolve_threshold(
        model, X_train_scaled, y_train
    )
    save_threshold(threshold, mean_prob, std_prob, source)

    # CHANGED Step 8 — Save feature importances (Task 3)
    save_feature_importances(model)

    print("\n  Training complete!  You can now run: python app.py")
    return model, scaler


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    run_training(algorithm="random_forest")