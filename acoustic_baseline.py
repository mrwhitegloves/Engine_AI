# ============================================================
# acoustic_baseline.py  —  Build a baseline acoustic signature
#                           from healthy engine audio files.
#
# HOW IT FITS IN THE PROJECT
# ──────────────────────────
#   This file is a SECOND, independent analysis layer on top
#   of the Random Forest classifier in train_model.py.
#
#   Random Forest  →  Learns PASS/FAIL from labelled examples
#   Baseline       →  Learns what "healthy" SOUNDS LIKE and
#                     measures how far any new recording is
#                     from that healthy fingerprint.
#
#   Both results are shown together in app.py.
#
# HOW TO RUN
# ──────────
#   python acoustic_baseline.py
#
#   It reads from:  dataset/normal/   (your healthy .wav files)
#   It saves to:    models/baseline.pkl
#                   models/baseline_features.csv
#                   models/avg_spectrogram.png
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import joblib

# ── Import shared settings from project config ───────────────
from config import (
    SAMPLE_RATE, HOP_LENGTH, N_FFT, N_MELS,
    NORMAL_DIR, MODEL_DIR,
    BASELINE_N_MFCC,
    BASELINE_PATH, BASELINE_CSV, BASELINE_PLOT
)

warnings.filterwarnings("ignore")


# =============================================================================
# AUDIO LOADING & PRE-PROCESSING
# =============================================================================

def get_wav_files(folder_path):
    """
    Scan a folder and return a sorted list of .wav file paths.

    Parameters
    ----------
    folder_path : str   Should point to dataset/normal/ by default.

    Returns
    -------
    list of str

    Raises
    ------
    FileNotFoundError  If the folder does not exist.
    ValueError         If no .wav files are found.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Audio folder not found: '{folder_path}'\n"
            f"Please create it and add healthy engine .wav files.\n"
            f"Tip: this is the same folder as your NORMAL_DIR in config.py"
        )

    wav_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".wav")
    ])

    if not wav_files:
        raise ValueError(
            f"No .wav files found in '{folder_path}'.\n"
            "Please add healthy engine audio recordings."
        )

    return wav_files


def load_audio_for_baseline(file_path):
    """
    Load a .wav file, resample to SAMPLE_RATE, convert to mono.
    Does NOT apply a duration cap — loads the full file for baseline.

    Returns (None, None) on error so the batch loop can skip bad files.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return audio, sr
    except Exception as err:
        print(f"    ⚠  Could not load '{os.path.basename(file_path)}': {err}")
        return None, None


def normalize_audio(audio):
    """
    Peak-normalize audio to [-1, +1].
    Prevents loud recordings from dominating the baseline average.
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        return audio
    return audio / peak


# =============================================================================
# FEATURE EXTRACTION  (41-feature set, independent of RF model features)
# =============================================================================

def extract_mfcc(audio, sr):
    """
    MFCC (Mel-Frequency Cepstral Coefficients).
    Captures the timbral texture / character of the engine sound.
    Uses BASELINE_N_MFCC = 20 (separate from the RF model's 40).

    Returns: np.ndarray shape (20,)
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=BASELINE_N_MFCC,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )
    return np.mean(mfcc, axis=1)    # mean over time → (20,)


def extract_chroma(audio, sr):
    """
    Chroma STFT — energy across 12 musical pitch classes.
    Useful for detecting harmonic anomalies (rattles, resonances).

    Returns: np.ndarray shape (12,)
    """
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return np.mean(chroma, axis=1)  # (12,)


def extract_spectral_contrast(audio, sr):
    """
    Spectral Contrast — amplitude difference between peaks and
    valleys across 7 frequency bands.
    High contrast in unexpected bands → possible mechanical fault.

    Returns: np.ndarray shape (7,)
    """
    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_bands=6              # n_bands+1 = 7 values returned
    )
    return np.mean(contrast, axis=1)    # (7,)


def extract_rms_energy(audio):
    """
    Root Mean Square Energy — overall loudness / power level.

    Returns: np.ndarray shape (1,)
    """
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    return np.array([np.mean(rms)])     # (1,)


def extract_zero_crossing_rate(audio):
    """
    Zero Crossing Rate — how often the waveform crosses zero.
    Higher ZCR = noisier sound (misfires, friction noise).

    Returns: np.ndarray shape (1,)
    """
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    return np.array([np.mean(zcr)])     # (1,)


def build_feature_vector(audio, sr):
    """
    Run all five extractors and concatenate into one flat vector.

    Layout
    ------
    [0:20]   MFCC means          (20 values)
    [20:32]  Chroma STFT means   (12 values)
    [32:39]  Spectral Contrast   ( 7 values)
    [39]     RMS Energy          ( 1 value)
    [40]     Zero Crossing Rate  ( 1 value)
    ─────────────────────────────────────────
    Total                        41 values

    Parameters
    ----------
    audio : np.ndarray   Normalized audio signal
    sr    : int

    Returns
    -------
    np.ndarray   Shape (41,)
    """
    mfcc      = extract_mfcc(audio, sr)
    chroma    = extract_chroma(audio, sr)
    contrast  = extract_spectral_contrast(audio, sr)
    rms       = extract_rms_energy(audio)
    zcr       = extract_zero_crossing_rate(audio)

    return np.concatenate([mfcc, chroma, contrast, rms, zcr])  # (41,)


def build_feature_names():
    """
    Human-readable column names matching build_feature_vector() layout.
    Used as CSV headers.

    Returns: list of 41 strings
    """
    names  = [f"mfcc_{i+1}"     for i in range(BASELINE_N_MFCC)]  # 20
    names += [f"chroma_{i+1}"   for i in range(12)]                # 12
    names += [f"contrast_{i+1}" for i in range(7)]                 #  7
    names += ["rms_energy"]                                         #  1
    names += ["zero_crossing_rate"]                                 #  1
    return names                                                    # 41


# =============================================================================
# DATASET PROCESSING
# =============================================================================

def process_all_files(wav_files):
    """
    Load, normalize, and extract features from every .wav file.

    Parameters
    ----------
    wav_files : list of str

    Returns
    -------
    feature_matrix : np.ndarray   Shape (n_valid_files, 41)
    file_names     : list of str
    mel_stack      : list of np.ndarray   (for the spectrogram plot)
    """
    feature_matrix = []
    file_names     = []
    mel_stack      = []
    total          = len(wav_files)

    print(f"\n  Processing {total} audio file(s)...\n")

    for index, file_path in enumerate(wav_files, start=1):
        name = os.path.basename(file_path)
        print(f"  [{index:>3}/{total}]  {name}", end="  ")

        # Load
        audio, sr = load_audio_for_baseline(file_path)
        if audio is None:
            print("→  SKIPPED (load error)")
            continue

        # Normalize
        audio = normalize_audio(audio)

        # Extract features
        try:
            fvec = build_feature_vector(audio, sr)
        except Exception as err:
            print(f"→  SKIPPED (feature error: {err})")
            continue

        # Collect mel-spectrogram for the bonus plot
        mel    = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_stack.append(mel_db)

        feature_matrix.append(fvec)
        file_names.append(name)
        print(f"→  OK  (shape: {fvec.shape})")

    return np.array(feature_matrix), file_names, mel_stack


# =============================================================================
# BASELINE CREATION
# =============================================================================

def compute_baseline(feature_matrix):
    """
    Compute the mean AND std feature vectors across all healthy samples.

    The mean  = acoustic fingerprint of a healthy engine.
    The std   = how much natural variation exists between healthy engines.
                Used later to compute how many standard deviations a
                test recording is away from the healthy fingerprint.

    Parameters
    ----------
    feature_matrix : np.ndarray   Shape (n_files, 41)

    Returns
    -------
    mean_vector : np.ndarray   Shape (41,)
    std_vector  : np.ndarray   Shape (41,)
    """
    mean_vector = np.mean(feature_matrix, axis=0)
    std_vector  = np.std(feature_matrix,  axis=0)
    # Prevent division by zero when computing z-score later
    std_vector  = np.where(std_vector < 1e-9, 1e-9, std_vector)
    return mean_vector, std_vector


def compute_intra_class_distances(feature_matrix, mean_vector, std_vector):
    """
    Compute how far EACH healthy sample is from the mean baseline.
    This tells us the natural spread within the healthy class, used
    to set realistic anomaly thresholds.

    Returns
    -------
    distances : np.ndarray   Shape (n_files,)
                Each value = normalized Euclidean distance from baseline.
    """
    distances = []
    for fvec in feature_matrix:
        # Normalize by std so all features have equal weight
        z_score  = (fvec - mean_vector) / std_vector
        distance = np.linalg.norm(z_score)
        distances.append(distance)
    return np.array(distances)


# =============================================================================
# OUTPUT: SAVE RESULTS
# =============================================================================

def save_baseline(mean_vector, std_vector, intra_distances, feature_names,
                  avg_mel_per_band=None):
    """
    Save the baseline as a dictionary to models/baseline.pkl.

    The saved dict contains everything needed at prediction time:
    - 'mean'             : mean feature vector of healthy class  (41,)
    - 'std'              : std feature vector of healthy class   (41,)
    - 'intra_mean'       : mean intra-class distance (natural spread)
    - 'intra_std'        : std of intra-class distances
    - 'feature_names'    : column labels for debugging
    - 'avg_mel_per_band' : mean dB energy per mel band  (N_MELS,)
                           Used by the worm graph in app.py.

    Parameters
    ----------
    mean_vector      : np.ndarray
    std_vector       : np.ndarray
    intra_distances  : np.ndarray
    feature_names    : list of str
    avg_mel_per_band : np.ndarray or None   Shape (N_MELS,)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    baseline_data = {
        "mean"             : mean_vector,
        "std"              : std_vector,
        "intra_mean"       : float(np.mean(intra_distances)),
        "intra_std"        : float(np.std(intra_distances)),
        "intra_max"        : float(np.max(intra_distances)),
        "feature_names"    : feature_names,
        "n_samples"        : len(intra_distances),
        # ── NEW: per-band mel energy for worm graph ───────────
        "avg_mel_per_band" : avg_mel_per_band if avg_mel_per_band is not None
                             else np.zeros(N_MELS),
    }
    joblib.dump(baseline_data, BASELINE_PATH)
    print(f"\n  💾  Baseline saved       →  {BASELINE_PATH}")


def save_feature_csv(feature_matrix, file_names, feature_names):
    """
    Write all individual feature vectors to a CSV for inspection in Excel.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.DataFrame(feature_matrix, columns=feature_names)
    df.insert(0, "filename", file_names)
    df.to_csv(BASELINE_CSV, index=False)
    print(f"  📄  Feature CSV saved     →  {BASELINE_CSV}")


def print_summary(feature_matrix, mean_vector, std_vector, intra_distances, file_names):
    """Print a human-readable summary of the baseline build."""
    n_files, n_features = feature_matrix.shape

    print("\n" + "=" * 60)
    print("  ACOUSTIC BASELINE  —  SUMMARY")
    print("=" * 60)
    print(f"  Files processed          :  {n_files}")
    print(f"  Feature vector length    :  {n_features}")
    print(f"  Intra-class dist (mean)  :  {np.mean(intra_distances):.4f}")
    print(f"  Intra-class dist (std)   :  {np.std(intra_distances):.4f}")
    print(f"  Intra-class dist (max)   :  {np.max(intra_distances):.4f}")
    print("─" * 60)
    print("  Baseline mean vector preview  (first 10 of 41 values):")
    preview = "  " + "  ".join(f"{v:+.4f}" for v in mean_vector[:10])
    print(preview)
    print("─" * 60)
    print("  Feature group ranges (mean):")
    print(f"    MFCC     [{mean_vector[0]:+.3f}  …  {mean_vector[19]:+.3f}]")
    print(f"    Chroma   [{mean_vector[20]:+.3f}  …  {mean_vector[31]:+.3f}]")
    print(f"    Contrast [{mean_vector[32]:+.3f}  …  {mean_vector[38]:+.3f}]")
    print(f"    RMS      [{mean_vector[39]:+.6f}]")
    print(f"    ZCR      [{mean_vector[40]:+.6f}]")
    print("=" * 60)


# =============================================================================
# BONUS: AVERAGE SPECTROGRAM PLOT
# =============================================================================

def plot_average_spectrogram(mel_stack):
    """
    Compute the element-wise mean of all mel-spectrograms and save as PNG.
    Saved to models/avg_spectrogram.png.
    """
    if not mel_stack:
        print("  ⚠  No spectrograms to plot.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Crop all spectrograms to the minimum length and stack
    min_frames = min(m.shape[1] for m in mel_stack)
    cropped    = [m[:, :min_frames] for m in mel_stack]
    avg_mel    = np.mean(np.stack(cropped, axis=0), axis=0)  # (N_MELS, frames)

    BG     = "#0F0F1A"
    ACCENT = "#E8E8F0"

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [3, 1]}
    )
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Average Mel-Spectrogram — Healthy Engine Baseline",
        color=ACCENT, fontsize=14, fontweight="bold", y=1.02
    )

    # Left panel: spectrogram image
    ax_spec = axes[0]
    ax_spec.set_facecolor(BG)
    img = librosa.display.specshow(
        avg_mel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel",
        ax=ax_spec, cmap="magma"
    )
    cbar = fig.colorbar(img, ax=ax_spec, format="%+2.0f dB", pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=ACCENT, labelsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=ACCENT)
    cbar.outline.set_edgecolor("#333355")
    ax_spec.set_title("Average Mel-Spectrogram (dB)", color=ACCENT, fontsize=11, pad=8)
    ax_spec.set_xlabel("Time (s)",      color=ACCENT, fontsize=10)
    ax_spec.set_ylabel("Mel Frequency", color=ACCENT, fontsize=10)
    ax_spec.tick_params(colors=ACCENT, labelsize=9)
    for sp in ax_spec.spines.values():
        sp.set_edgecolor("#333355")

    # Right panel: per-band mean energy bar
    ax_bar = axes[1]
    ax_bar.set_facecolor(BG)
    band_energy = np.mean(avg_mel, axis=1)
    y_pos       = np.arange(len(band_energy))
    norm_e      = (band_energy - band_energy.min()) / (np.ptp(band_energy) + 1e-9)  # FIXED: .ptp() removed in NumPy 2.0, use np.ptp()
    ax_bar.barh(y_pos, band_energy, color=plt.cm.magma(norm_e), height=1.0, edgecolor="none")
    ax_bar.set_title("Mean Energy\nper Mel Band", color=ACCENT, fontsize=10, pad=8)
    ax_bar.set_xlabel("dB",       color=ACCENT, fontsize=9)
    ax_bar.set_ylabel("Mel Band", color=ACCENT, fontsize=9)
    ax_bar.tick_params(colors=ACCENT, labelsize=8)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor("#333355")
    ax_bar.grid(axis="x", color="#333355", alpha=0.5, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(BASELINE_PLOT, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  📊  Spectrogram plot saved →  {BASELINE_PLOT}")


# =============================================================================
# MAIN PIPELINE  (called by app.py AND runnable directly)
# =============================================================================

def build_acoustic_baseline(audio_folder=NORMAL_DIR, log_fn=print):
    """
    End-to-end baseline building pipeline.

    Can be called two ways:
      1. Directly:   python acoustic_baseline.py
      2. From app.py:  build_acoustic_baseline(log_fn=some_logger)

    Parameters
    ----------
    audio_folder : str       Folder of healthy .wav files.
                             Defaults to dataset/normal/ from config.
    log_fn       : callable  Where to send log messages.
                             Use print (default) or a Gradio logger.

    Returns
    -------
    mean_vector : np.ndarray   Shape (41,)   The baseline fingerprint.
    std_vector  : np.ndarray   Shape (41,)   Per-feature variation.

    Raises
    ------
    FileNotFoundError / ValueError / RuntimeError on bad input.
    """
    log_fn("\n" + "=" * 55)
    log_fn("  ENGINE ACOUSTIC BASELINE BUILDER")
    log_fn("=" * 55)

    # Step 1 — find files
    wav_files = get_wav_files(audio_folder)
    log_fn(f"\n  Found {len(wav_files)} .wav file(s) in '{audio_folder}'")

    # Step 2 — process all files
    feature_matrix, file_names, mel_stack = process_all_files(wav_files)

    if len(feature_matrix) == 0:
        raise RuntimeError(
            "No files processed successfully.\n"
            "Check that the folder contains valid .wav recordings."
        )

    # Step 3 — compute baseline statistics
    mean_vector, std_vector = compute_baseline(feature_matrix)
    intra_distances = compute_intra_class_distances(
        feature_matrix, mean_vector, std_vector
    )

    feature_names = build_feature_names()

    # Step 4 — compute per-band mel energy for the worm graph
    #           Crop all mel spectrograms to the shortest, average over
    #           time → one energy value per mel band (shape: N_MELS,)
    if mel_stack:
        min_frames       = min(m.shape[1] for m in mel_stack)
        cropped          = [m[:, :min_frames] for m in mel_stack]
        avg_mel_matrix   = np.mean(np.stack(cropped, axis=0), axis=0)  # (N_MELS, frames)
        avg_mel_per_band = np.mean(avg_mel_matrix, axis=1)              # (N_MELS,)
    else:
        avg_mel_per_band = np.zeros(N_MELS)

    # Step 5 — save all outputs
    log_fn("")
    save_baseline(mean_vector, std_vector, intra_distances, feature_names,
                  avg_mel_per_band=avg_mel_per_band)
    save_feature_csv(feature_matrix, file_names, feature_names)
    plot_average_spectrogram(mel_stack)

    # Step 5 — print summary
    print_summary(feature_matrix, mean_vector, std_vector, intra_distances, file_names)

    log_fn(f"\n  ✅  Baseline build complete!  ({len(feature_matrix)} files used)")
    log_fn(f"  Saved to: {BASELINE_PATH}")
    return mean_vector, std_vector


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    build_acoustic_baseline()