# ============================================================
# train_model.py  —  Train the AI model and save it to disk
#
# CHANGES vs previous version (all marked # CHANGED):
#   Task 1  — compute_adaptive_threshold()  : finds optimal PASS threshold
#              from the actual training-data normal-class probabilities.
#              Saves to models/threshold.pkl so predict.py can load it.
#   Task 3  — save_feature_importances()    : saves RF feature_importances_
#              to models/feature_importances.pkl for weighted z-score scoring.
#   run_training()                          : calls both functions above.
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
    MODEL_DIR                                          # CHANGED — needed for new paths
)

# CHANGED — paths for new saved artefacts
THRESHOLD_PATH     = "models/threshold.pkl"           # CHANGED
IMPORTANCES_PATH   = "models/feature_importances.pkl" # CHANGED


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
# 6.  ADAPTIVE THRESHOLD  (NEW — Task 1)
# ────────────────────────────────────────────────────────────

def compute_adaptive_threshold(model, X_train_scaled, y_train):
    """
    CHANGED (Task 1) — Compute the optimal PASS threshold from the
    actual distribution of normal-class probabilities in the training data.

    Problem being solved:
        With a small dataset the RF model naturally produces lower
        probabilities.  A hardcoded 90% threshold will reject many truly
        normal engines because the model only outputs e.g. 65–75% for normals.

    Method — 5th-percentile rule:
        1. Get prob_normal for every KNOWN-NORMAL training sample.
        2. Set threshold = 5th percentile of those probabilities.
        3. This guarantees that 95% of the training normal samples pass,
           while the other 5% (outliers in the training set) still may not.
        4. Clamp to [0.40, 0.85] so the threshold is always sensible.

    Returns
    -------
    threshold  : float  The adaptive PASS threshold (0–1)
    mean_prob  : float  Mean normal-class probability (for reporting)
    std_prob   : float  Std of normal-class probabilities
    """
    # Get prob_normal for all training samples
    probs = model.predict_proba(X_train_scaled)

    # Keep only the known-normal samples
    normal_mask  = (y_train == 0)
    normal_probs = probs[normal_mask, 0]      # prob_normal for true normals

    if len(normal_probs) == 0:
        return 0.65, 0.65, 0.0               # fallback if no normal samples

    mean_prob = float(np.mean(normal_probs))
    std_prob  = float(np.std(normal_probs))

    # 5th percentile — 95 % of training normals will be above this
    p5 = float(np.percentile(normal_probs, 5))

    # Clamp to a sensible range
    threshold = float(np.clip(p5, 0.40, 0.85))

    print(f"\n  Adaptive threshold computed from {int(normal_mask.sum())} normal training samples:")
    print(f"    Mean prob_normal  : {mean_prob:.4f}")
    print(f"    Std  prob_normal  : {std_prob:.4f}")
    print(f"    5th percentile    : {p5:.4f}")
    print(f"    Clamped threshold : {threshold:.4f}  ({threshold*100:.1f}%)")
    return threshold, mean_prob, std_prob


def save_adaptive_threshold(threshold, mean_prob, std_prob):
    """
    CHANGED (Task 1) — Persist the adaptive threshold to disk so that
    predict.py can load it without re-running training.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    data = {
        "threshold" : threshold,
        "mean_prob" : mean_prob,
        "std_prob"  : std_prob,
    }
    joblib.dump(data, THRESHOLD_PATH)
    print(f"  Threshold saved -> {THRESHOLD_PATH}  (value: {threshold:.4f})")


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

    # CHANGED Step 7 — Compute and save adaptive threshold (Task 1)
    threshold, mean_prob, std_prob = compute_adaptive_threshold(
        model, X_train_scaled, y_train
    )
    save_adaptive_threshold(threshold, mean_prob, std_prob)

    # CHANGED Step 8 — Save feature importances (Task 3)
    save_feature_importances(model)

    print("\n  Training complete!  You can now run: python app.py")
    return model, scaler


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    run_training(algorithm="random_forest")