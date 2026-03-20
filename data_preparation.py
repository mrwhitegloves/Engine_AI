# ============================================================
# data_preparation.py  —  Scan dataset folders and build the
#                          feature matrix X and label array y
# ============================================================

import os
import glob
import numpy as np
from feature_extraction import extract_features
from config import NORMAL_DIR, ABNORMAL_DIR, MODEL_DIR


# ────────────────────────────────────────────────────────────
# 1.  CREATE REQUIRED FOLDERS IF THEY DON'T EXIST
# ────────────────────────────────────────────────────────────
def create_project_folders():
    """
    Create dataset/normal, dataset/abnormal, and models folders.
    Safe to call multiple times — won't overwrite existing files.
    """
    for folder in [NORMAL_DIR, ABNORMAL_DIR, MODEL_DIR]:
        os.makedirs(folder, exist_ok=True)
    print("✅ Project folders are ready:")
    print(f"   Normal audio   → {NORMAL_DIR}")
    print(f"   Abnormal audio → {ABNORMAL_DIR}")
    print(f"   Saved models   → {MODEL_DIR}")


# ────────────────────────────────────────────────────────────
# 2.  FIND ALL AUDIO FILES IN A FOLDER
# ────────────────────────────────────────────────────────────
def get_audio_files(directory):
    """
    Scan a directory for supported audio files.

    Supported formats: .wav, .mp3, .flac, .ogg, .m4a

    Returns
    -------
    files : list of str   Absolute paths to audio files
    """
    supported = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]
    files = []
    for pattern in supported:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    return sorted(files)


# ────────────────────────────────────────────────────────────
# 3.  PROCESS ONE AUDIO FILE AND PRINT PROGRESS
# ────────────────────────────────────────────────────────────
def process_audio_file(file_path, label, index, total):
    """
    Extract features from a single audio file with progress output.

    Parameters
    ----------
    file_path : str
    label     : int   0 = Normal, 1 = Abnormal
    index     : int   Current file number (for progress display)
    total     : int   Total files in this group

    Returns
    -------
    features : np.ndarray or None (if file is corrupted)
    """
    label_name = "Normal" if label == 0 else "Abnormal"
    try:
        features = extract_features(file_path)
        print(f"  [{index}/{total}] ✅ {label_name} — {os.path.basename(file_path)}")
        return features
    except Exception as err:
        print(f"  [{index}/{total}] ❌ Skipped (error) — {os.path.basename(file_path)}: {err}")
        return None


# ────────────────────────────────────────────────────────────
# 4.  BUILD FULL DATASET  (X, y)
# ────────────────────────────────────────────────────────────
def prepare_dataset():
    """
    Scan both dataset folders, extract features from every audio
    file, and return feature matrix X and label array y.

    Labels
    ------
    0 = Normal (PASS)
    1 = Abnormal (FAIL)

    Returns
    -------
    X : np.ndarray  shape (n_samples, n_features)
    y : np.ndarray  shape (n_samples,)
    """
    normal_files   = get_audio_files(NORMAL_DIR)
    abnormal_files = get_audio_files(ABNORMAL_DIR)

    print("\n" + "="*55)
    print("  ENGINE AUDIO DATASET PREPARATION")
    print("="*55)
    print(f"  Normal   recordings found : {len(normal_files)}")
    print(f"  Abnormal recordings found : {len(abnormal_files)}")
    print("="*55)

    # Guard: need at least some files in each folder
    if len(normal_files) == 0:
        raise FileNotFoundError(
            f"No audio files in '{NORMAL_DIR}'. "
            "Please add normal (good) engine recordings."
        )
    if len(abnormal_files) == 0:
        raise FileNotFoundError(
            f"No audio files in '{ABNORMAL_DIR}'. "
            "Please add abnormal (faulty) engine recordings."
        )

    X = []
    y = []

    # ── Normal files ────────────────────────────────────────
    print("\n📁 Processing NORMAL engine sounds ...")
    for i, file_path in enumerate(normal_files, start=1):
        features = process_audio_file(file_path, label=0, index=i, total=len(normal_files))
        if features is not None:
            X.append(features)
            y.append(0)

    # ── Abnormal files ──────────────────────────────────────
    print("\n📁 Processing ABNORMAL engine sounds ...")
    for i, file_path in enumerate(abnormal_files, start=1):
        features = process_audio_file(file_path, label=1, index=i, total=len(abnormal_files))
        if features is not None:
            X.append(features)
            y.append(1)

    X = np.array(X)
    y = np.array(y)

    print(f"\n✅ Dataset ready — {len(X)} total samples, {X.shape[1]} features each")
    print(f"   Normal: {np.sum(y == 0)}  |  Abnormal: {np.sum(y == 1)}")
    return X, y


# ────────────────────────────────────────────────────────────
# 5.  QUICK HEALTH CHECK
# ────────────────────────────────────────────────────────────
def check_dataset_health():
    """
    Print a quick summary of what is in the dataset folders.
    Useful before training to verify you have enough files.
    """
    normal_files   = get_audio_files(NORMAL_DIR)
    abnormal_files = get_audio_files(ABNORMAL_DIR)

    print("\n📊 DATASET HEALTH CHECK")
    print(f"  Normal   folder : {len(normal_files)} files")
    print(f"  Abnormal folder : {len(abnormal_files)} files")

    if len(normal_files) < 5 or len(abnormal_files) < 5:
        print("\n⚠️  WARNING: Very few audio files detected.")
        print("   For a reliable AI model, aim for at least 20+ files per class.")
    else:
        print("\n✅ Dataset looks good for training!")


# Run health check if this script is executed directly
if __name__ == "__main__":
    create_project_folders()
    check_dataset_health()
