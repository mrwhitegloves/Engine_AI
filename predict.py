# ============================================================
# predict.py  —  Run prediction on a single audio file and
#                generate visualisation plots for the UI
#
# CHANGES vs previous version (all marked # CHANGED):
#   Task 1  — load_adaptive_threshold()   : loads threshold.pkl from training
#   Task 3  — load_feature_importances()  : loads feature_importances_.pkl
#   Task 2  — compute_weighted_zscore_vote(): z-score vote with importance weights
#   Task 6  — compute_ensemble_status()   : 3-detector voting (RF+baseline+zscore)
#   Task 7  — predict_engine_status()     : three-zone PASS/WARNING/FAIL
#   Task 7  — format_result_text()        : shows WARNING zone and threshold
# ============================================================

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
from ann_model import predict as ann_predict

from feature_extraction import (
    extract_features, load_audio,
    get_spectrogram_data, get_mel_spectrogram_data,
    extract_mfcc_features
)
from train_model import load_model
from config import (
    SAMPLE_RATE, HOP_LENGTH, N_FFT,
    BASELINE_PATH, BASELINE_WARN_SIGMA, BASELINE_FAIL_SIGMA,
    PASS_THRESHOLD,                    # CHANGED — three-zone thresholds
    ENSEMBLE_FAIL_VOTES,                               # CHANGED — ensemble voting
    WEIGHT_MFCC, WEIGHT_SPECTRAL, WEIGHT_OTHER         # CHANGED — feature weights
)


# ────────────────────────────────────────────────────────────
# COLOUR PALETTE  (consistent across all plots)
# ────────────────────────────────────────────────────────────
PASS_COLOR = "#2ECC71"    # green
FAIL_COLOR = "#E74C3C"    # red 
WAVE_COLOR = "#3498DB"    # blue
BG_COLOR   = "#1A1A2E"    # dark navy
GRID_COLOR = "#2C2C54"
TEXT_COLOR = "#ECF0F1"

# ── Paths for new artefacts saved by train_model.py ──────────
THRESHOLD_PATH   = "models/threshold.pkl"              # CHANGED
IMPORTANCES_PATH = "models/feature_importances.pkl"    # CHANGED


# ────────────────────────────────────────────────────────────
# NEW HELPERS  (Task 1, 2, 3, 6)
# ────────────────────────────────────────────────────────────

def load_adaptive_threshold():
    """
    CHANGED (Task 1) — Load the adaptive PASS threshold that was computed
    from the training data by train_model.py.

    Falls back to config.PASS_THRESHOLD if threshold.pkl does not exist
    (e.g. model was trained before this update).

    Returns
    -------
    pass_threshold : float   Normal Score >= this -> PASS
    warn_threshold : float   Normal Score >= this -> WARNING (below -> FAIL)
    source         : str     "adaptive" or "config_default"
    """
    if os.path.exists(THRESHOLD_PATH):
        data           = joblib.load(THRESHOLD_PATH)
        pass_thr       = float(data["threshold"])
        return pass_thr, "adaptive"
    # Fallback to config defaults
    return PASS_THRESHOLD, "config_default"


def load_feature_importances():
    """
    CHANGED (Task 3) — Load RF feature importances saved by train_model.py.

    Returns None if the file does not exist (graceful degradation:
    all features get equal weight in that case).
    """
    if os.path.exists(IMPORTANCES_PATH):
        return joblib.load(IMPORTANCES_PATH)   # np.ndarray shape (226,)
    return None


def compute_weighted_zscore_vote(file_path, importances=None):
    """
    CHANGED (Task 2 & 3) — Z-score based vote using feature importance weights.

    For each of the 226 features, compute how many standard deviations
    the current engine is from the training-mean (stored in the scaler).
    Then compute a weighted anomaly score:
        - MFCC features  get weight WEIGHT_MFCC   (50%)
        - Spectral/RMS   get weight WEIGHT_SPECTRAL (30%)
        - Chroma/ZCR/Mel get weight WEIGHT_OTHER    (20%)
    Further, if feature_importances are available, multiply each feature's
    z-score by its importance so that low-importance noisy features cannot
    alone trigger a false alarm.

    Returns
    -------
    vote      : str     "PASS" / "WARNING" / "FAIL"
    score     : float   Weighted anomaly score (higher = more abnormal)
    """
    try:
        model, scaler = load_model()
        features      = extract_features(file_path)          # (226,)

        # Z-score = how many std from the scaler's training mean
        # scaler.mean_ and scaler.scale_ were learned on the training data
        z_scores = (features - scaler.mean_) / (scaler.scale_ + 1e-9)  # (226,)

        # ── Build feature group weight vector ─────────────────
        n = len(z_scores)
        weights = np.ones(n, dtype=np.float32)

        # Feature layout: [0:80]=MFCC, [80:84]=spectral, [84:96]=chroma,
        #                  [96:98]=rms, [98:226]=mel
        weights[0:80]   *= WEIGHT_MFCC      # MFCC mean+std (Task 3 weight)
        weights[80:84]  *= WEIGHT_SPECTRAL  # spectral features
        weights[84:96]  *= WEIGHT_OTHER     # chroma
        weights[96:98]  *= WEIGHT_SPECTRAL  # RMS
        weights[98:226] *= WEIGHT_OTHER     # mel mean (128 bands)

        # ── Apply feature importances if available ─────────────
        if importances is not None and len(importances) == n:
            # Normalize importances to mean=1 so they act as multipliers
            norm_imp = importances / (importances.mean() + 1e-9)
            weights  = weights * norm_imp                    # CHANGED — importance weighting

        # ── Weighted anomaly score ─────────────────────────────
        # Use absolute z-scores: large positive OR negative deviation = anomaly
        abs_z         = np.abs(z_scores)
        weighted_score = float(np.sum(abs_z * weights) / (np.sum(weights) + 1e-9))

        # ── Vote thresholds ────────────────────────────────────
        # These are empirically chosen: a well-normalized healthy engine
        # should have a weighted z-score below 1.5
        if weighted_score < 2.0:
            vote = "PASS"
        else:
            vote = "FAIL"

        return vote, weighted_score

    except Exception:
        return "PASS", 0.0              # if anything fails, don't penalise


def compute_ensemble_status(prob_normal, baseline_risk_level, zscore_vote, file_path):
    """
    CHANGED (Task 6 & 10) — Ensemble voting from 3 independent detectors.

    Detectors
    ---------
    1. RF model      : PASS / WARNING / FAIL  (from prob_normal + adaptive threshold)
    2. Baseline sigma: OK -> PASS, WARNING -> FAIL, ALERT -> FAIL
    3. Z-score       : PASS / WARNING / FAIL  (from compute_weighted_zscore_vote)

    Voting rule (Task 6 Confidence Rule):
        - Count how many detectors vote FAIL
        - Count how many detectors vote WARNING or worse
        - FAIL only if fail_votes >= ENSEMBLE_FAIL_VOTES  (default 2)
        - WARNING if any single detector says WARNING/FAIL but not enough for FAIL
        - PASS if all detectors say PASS

    This directly prevents a single uncertain detector from causing a
    false positive — at least 2 out of 3 must agree before engine is FAIL.

    Returns
    -------
    final_status   : str    "PASS" / "FAIL"
    explanation    : str    Brief reason for the decision
    """
    pass_thr,_ = load_adaptive_threshold()

    # ── Detector 1: RF model ───────────────────────────────────
    if prob_normal >= pass_thr:
        rf_vote = "PASS"
    else:
        rf_vote = "FAIL"

    # ── Detector 2: Baseline sigma ─────────────────────────────
    baseline_map = {"OK": "PASS", "WARNING": "FAIL", "ALERT": "FAIL",
                    "N/A": "PASS"}
    baseline_vote = baseline_map.get(baseline_risk_level, "PASS")

    # ── Detector 3: Z-score vote (already computed) ────────────
    zscore_v = zscore_vote   # "PASS" / "WARNING" / "FAIL"

    votes = [rf_vote, baseline_vote, zscore_v]
    try:
        ann_status, _, _ = ann_predict(file_path)
        votes.append(ann_status)
    except:
        pass
    fail_count    = votes.count("FAIL")

    if fail_count >= ENSEMBLE_FAIL_VOTES:          # >= 2 detectors say FAIL
        final_status = "FAIL"
        explanation  = (f"[RF={rf_vote}, Baseline={baseline_vote}, "
                        f"ZScore={zscore_v}] — {fail_count}/3 FAIL votes")
    else:                                          # most detectors say PASS
        final_status = "PASS"
        explanation  = (f"[RF={rf_vote}, Baseline={baseline_vote}, "
                        f"ZScore={zscore_v}] — majority PASS")

    return final_status, explanation


# ────────────────────────────────────────────────────────────
# 1.  CORE PREDICTION  (Random Forest + adaptive threshold)
# ────────────────────────────────────────────────────────────
def predict_engine_status(file_path):
    """
    Load the trained model, extract features, apply adaptive threshold,
    and return PASS / FAIL with probabilities.

    CHANGED (Task 1 & 7):
    - Loads adaptive threshold from models/threshold.pkl (falls back to config)
    - Uses THREE zones: PASS / FAIL instead of binary PASS/FAIL
    - PASS  = Normal Score >= pass_threshold  (adaptive, ~95th pct of training normals)
    - FAIL  = ELSE FALSE

    Returns
    -------
    status        : str    "PASS" / "FAIL"
    confidence    : float  Confidence % (0–100)
    prediction    : int    0=Normal, 1=Abnormal  (raw model vote, unchanged)
    prob_normal   : float  Probability of being normal (0–1)
    prob_abnormal : float  Probability of being abnormal (0–1)
    pass_threshold: float  The threshold used (for display in report)
    """
    model, scaler = load_model()

    features        = extract_features(file_path)
    features_scaled = scaler.transform(features.reshape(1, -1))

    prediction    = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    prob_normal   = probabilities[0]
    prob_abnormal = probabilities[1]

    # CHANGED — load adaptive threshold from training data
    pass_thr, thr_source = load_adaptive_threshold()

    # CHANGED — three-zone decision
    if prob_normal >= pass_thr:
        status = "PASS"
    else:
        status = "FAIL"

    # Confidence = certainty of the actual status
    if status == "PASS":
        confidence = prob_normal * 100
    else:
        confidence = prob_abnormal * 100

    return status, confidence, prediction, prob_normal, prob_abnormal, pass_thr


# ────────────────────────────────────────────────────────────
# 2.  BASELINE COMPARISON  (uses acoustic_baseline.py)
# ────────────────────────────────────────────────────────────

def load_baseline():
    """Load baseline dict from models/baseline.pkl."""
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(
            f"Baseline not found at '{BASELINE_PATH}'.\n"
            "Build it first using the Baseline Builder tab in the app,\n"
            "or run: python acoustic_baseline.py"
        )
    return joblib.load(BASELINE_PATH)

# ────────────────────────────────────────────────────────────
# 3.  PLOT: WORM GRAPH  (V3 Update: single panel, time-axis)
# ────────────────────────────────────────────────────────────

def plot_worm_graph(file_path):
    """
    V3 Update: Single-panel cricket-style worm graph.

    Requested change vs V2 (4-panel):
      - Back to ONE panel (matches the reference screenshot style)
      - X-axis = TIME IN SECONDS  (was: mel band index 0-127)
      - Y-axis = Cumulative Energy (same cricket-score metaphor)

    How the time-based worm works
    ─────────────────────────────
    1. Compute mel spectrogram for the current engine  → (n_mels, n_frames)
    2. Average over mel bands per frame                → (n_frames,)  "energy at each moment"
    3. Cumulative sum over frames                      → always-rising worm curve
    4. Baseline = constant energy per frame (mean of baseline mel bands)
       gives a perfectly straight reference line — any deviation = anomaly

    Anomaly detection (kept from V2)
    ────────────────────────────────
    - z_score per frame = (current_frame_energy - baseline_mean) / baseline_std_est
    - Flag frames where |z| > ZSCORE_THRESH (1.5)
    - Show as red dots on the current engine line + red vertical shading

    Visual style (matches reference screenshot)
    ───────────────────────────────────────────
    Blue  + circle markers  = Healthy Baseline (straight cumulative line)
    Grey  + square markers  = Current Engine
    Red filled circles      = Anomaly time moments
    Red vertical shading    = Anomaly regions
    """
    from acoustic_baseline import load_audio_for_baseline, normalize_audio
    from config import N_MELS as _N_MELS

    # ── Load baseline ─────────────────────────────────────────
    baseline_data = load_baseline()
    baseline_mel  = baseline_data.get("avg_mel_per_band", None)

    # ── Load + normalize current audio ────────────────────────
    audio, sr = load_audio_for_baseline(file_path)
    if audio is None:
        return None
    audio   = normalize_audio(audio)
    n_bands = _N_MELS   # 128

    # ── V3 Update: mel spectrogram keeping TIME axis intact ───────────────────
    # We keep the full (n_mels, n_frames) matrix so we can plot over TIME.
    # ref=1.0 is kept from V2 — absolute reference prevents false overlap.
    mel     = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_bands, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db  = librosa.power_to_db(mel, ref=1.0)          # V3 Update: absolute ref (from V2)

    # V3 Update: mean over mel bands → one energy value per TIME FRAME
    # Shape: (n_frames,)  — "how loud is the engine at each moment in time"
    current_energy_frames = np.mean(mel_db, axis=0)       # V3 Update: energy per time frame

    # V3 Update: time axis in real seconds
    n_frames   = current_energy_frames.shape[0]
    time_axis  = librosa.frames_to_time(                  # V3 Update: convert frames → seconds
        np.arange(n_frames), sr=sr, hop_length=HOP_LENGTH
    )
    total_time = float(time_axis[-1])

    # ── V3 Update: baseline reference energy ──────────────────────────────────
    # The baseline pkl stores avg_mel_per_band (shape 128,) = mean dB per band.
    # Mean of this vector = expected mean energy per time frame.
    # This gives a CONSTANT energy per frame → straight cumulative line
    # (a healthy engine runs at a steady level, like scoring at a constant rate).
    if baseline_mel is not None and not np.all(baseline_mel == 0):
        baseline_frame_mean = float(np.mean(baseline_mel))    # V3 Update: scalar mean energy
    else:
        baseline_frame_mean = float(np.mean(current_energy_frames))  # fallback

    # V3 Update: baseline std estimated from intra_std in the pkl
    intra_std            = float(baseline_data.get("intra_std", 1.0))
    baseline_frame_std   = abs(baseline_frame_mean) * (intra_std / 10.0) + 1e-6  # V3 Update

    # V3 Update: baseline worm = perfectly straight line (constant energy/frame)
    baseline_energy_frames = np.full(n_frames, baseline_frame_mean)              # V3 Update

    # ── V3 Update: cumulative worm curves (over time, not over mel bands) ─────
    # Shift so both curves start at 0 (dB values can be negative)
    offset           = min(baseline_frame_mean, current_energy_frames.min())
    b_shifted        = np.clip(baseline_energy_frames - offset, 0, None)
    c_shifted        = np.clip(current_energy_frames  - offset, 0, None)

    b_cum            = np.cumsum(b_shifted)
    c_cum            = np.cumsum(c_shifted)

    # V3 Update: scale from baseline only (fix from V2 — prevents forced overlap)
    scale_f          = 300.0 / (b_cum[-1] + 1e-9)         # V3 Update: baseline-only scale
    baseline_worm    = b_cum * scale_f
    current_worm     = c_cum * scale_f                     # V3 Update: current uses baseline scale

    # ── V3 Update: anomaly detection on TIME frames ───────────────────────────
    ZSCORE_THRESH    = 1.5                                 # V3 Update: 1.5σ (kept from V2)
    z_per_frame      = (current_energy_frames - baseline_frame_mean) / baseline_frame_std
    anomaly_frames   = np.where(np.abs(z_per_frame) > ZSCORE_THRESH)[0]          # V3 Update

    # V3 Update: anomaly times in seconds (for vertical shading)
    anomaly_times    = time_axis[anomaly_frames]           # V3 Update: real time positions

    # ── Marker step — one dot per ~0.5 seconds (like cricket overs) ──────────
    # V3 Update: markers spaced in time, not in band index
    marker_step      = max(1, n_frames // 20)             # V3 Update: ~20 markers total
    mark_idx         = np.arange(0, n_frames, marker_step)

    # ═══════════════════════════════════════════════════════════
    # V3 Update: SINGLE PANEL figure (was 4-panel in V2)
    # ═══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(13, 5))               # V3 Update: single axis
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # V3 Update: red vertical shading at anomaly TIME positions
    for t in anomaly_times:
        ax.axvspan(t - 0.02, t + 0.02,                   # V3 Update: time-width shading
                   color=FAIL_COLOR, alpha=0.20, zorder=1)

    # ── Healthy baseline line — blue + circle markers ─────────
    ax.plot(
        time_axis, baseline_worm,
        color="#3498DB", linewidth=2.2, alpha=0.95,
        label="Good Engine (Baseline)", zorder=3
    )
    ax.plot(
        time_axis[mark_idx], baseline_worm[mark_idx],
        "o", color="#3498DB", markersize=7,
        markerfacecolor="#3498DB", markeredgecolor=BG_COLOR,
        markeredgewidth=1.2, zorder=4
    )

    # ── Current engine line — grey + square markers ───────────
    ax.plot(
        time_axis, current_worm,
        color="#95A5A6", linewidth=2.2, alpha=0.95,
        label="Current Engine", zorder=3
    )
    ax.plot(
        time_axis[mark_idx], current_worm[mark_idx],
        "s", color="#95A5A6", markersize=7,
        markerfacecolor="#95A5A6", markeredgecolor=BG_COLOR,
        markeredgewidth=1.2, zorder=4
    )

    # ── Red anomaly dots on the current engine worm ───────────
    if len(anomaly_frames) > 0:
        ax.plot(
            anomaly_times, current_worm[anomaly_frames],
            "o", color=FAIL_COLOR, markersize=11, zorder=5,
            label=f"Anomaly Moment ({len(anomaly_frames)})",
            markerfacecolor=FAIL_COLOR,
            markeredgecolor="#FFD0D0", markeredgewidth=1.5
        )

    # ── Horizontal reference grid lines (like cricket score grid) ────────────
    for y_val in [50, 100, 150, 200, 250, 300]:
        ax.axhline(y_val, color=GRID_COLOR, linewidth=0.5, alpha=0.5, zorder=0)

    # ── Axis labels, title, limits ────────────────────────────
    ax.set_title(
        "Worm Graph  |  Cumulative Energy vs Time: Healthy Baseline vs Current Engine",
        color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=12
    )
    # V3 Update: X-axis is TIME IN SECONDS (was mel band index)
    ax.set_xlabel("Time (seconds)",                       # V3 Update: time axis label
                  color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Cumulative Energy (scaled 0-300 from baseline)",
                  color=TEXT_COLOR, fontsize=11)
    ax.set_xlim(0, total_time)                            # V3 Update: time range
    ax.set_ylim(0, max(baseline_worm.max(), current_worm.max()) * 1.10)
    ax.tick_params(colors=TEXT_COLOR)

    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COLOR)

    ax.legend(
        loc="upper left", framealpha=0.25,
        facecolor=BG_COLOR, edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR, fontsize=10
    )

    # ── Anomaly badge (bottom-right) ──────────────────────────
    n_anom  = len(anomaly_frames)
    badge_c = FAIL_COLOR if n_anom > 0 else PASS_COLOR
    badge_t = (f"[!] {n_anom} anomaly moment(s)  |z| > {ZSCORE_THRESH} sigma"
               if n_anom > 0 else "[OK] No anomaly moments detected")
    ax.text(
        0.98, 0.05, badge_t,
        transform=ax.transAxes, color=badge_c,
        fontsize=10, fontweight="bold", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor=BG_COLOR, edgecolor=badge_c, alpha=0.88)
    )

    fig.tight_layout()
    plt.close("all")   # V3 Update: prevent Gradio memory leak
    return fig



# ────────────────────────────────────────────────────────────
# 4.  PLOT: ANOMALY INDICATORS  (per-feature red/green bars)
# ────────────────────────────────────────────────────────────

def plot_anomaly_indicators(file_path):
    """
    Per-feature z-score bar chart with colour-coded anomaly indicators.

    Each of the 41 baseline features gets a bar showing how many
    standard deviations the current engine is away from the healthy mean.

    Colours
    ───────
    🟢 Green  — |z| < BASELINE_WARN_SIGMA   (within normal variation)
    🔴 Red    — |z| ≥ BASELINE_FAIL_SIGMA   (alert — strong anomaly)

    Red dashed threshold lines and 🔴 emoji markers highlight alerts.

    Parameters
    ----------
    file_path : str   Path to the current engine audio file.

    Returns
    -------
    fig : matplotlib.figure.Figure  or  None on error.
    """
    from acoustic_baseline import (
        load_audio_for_baseline, normalize_audio,
        build_feature_vector, build_feature_names
    )

    # ── Load baseline and test features ──────────────────────
    baseline  = load_baseline()
    mean_vec  = baseline["mean"]
    std_vec   = baseline["std"]
    feat_names = baseline.get("feature_names", build_feature_names())

    audio, sr = load_audio_for_baseline(file_path)
    if audio is None:
        return None
    audio     = normalize_audio(audio)
    test_fvec = build_feature_vector(audio, sr)   # (41,)

    # ── Compute per-feature z-scores ─────────────────────────
    z_scores = (test_fvec - mean_vec) / std_vec    # (41,)
    n        = len(z_scores)

    # ── Assign colour per bar ─────────────────────────────────
    bar_colors = []
    for z in z_scores:
        if abs(z) >= BASELINE_FAIL_SIGMA:
            bar_colors.append(FAIL_COLOR)          # red alert
        else:
            bar_colors.append(PASS_COLOR)          # green OK

    n_alert = bar_colors.count(FAIL_COLOR)

    x = np.arange(n)

    # ── Figure ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Background shading for threshold zones
    y_abs_max = max(np.abs(z_scores).max() * 1.15, BASELINE_FAIL_SIGMA + 1.0)
    ax.axhspan(-BASELINE_WARN_SIGMA,  BASELINE_WARN_SIGMA,
               color=PASS_COLOR, alpha=0.05, zorder=0, label="_nolegend_")
    ax.axhspan( BASELINE_WARN_SIGMA,  BASELINE_FAIL_SIGMA,
               color=FAIL_COLOR, alpha=0.07, zorder=0, label="_nolegend_")
    ax.axhspan(-BASELINE_FAIL_SIGMA, -BASELINE_WARN_SIGMA,
               color=FAIL_COLOR, alpha=0.07, zorder=0, label="_nolegend_")
    ax.axhspan( BASELINE_FAIL_SIGMA,  y_abs_max,
               color=FAIL_COLOR, alpha=0.07, zorder=0, label="_nolegend_")
    ax.axhspan(-y_abs_max, -BASELINE_FAIL_SIGMA,
               color=FAIL_COLOR, alpha=0.07, zorder=0, label="_nolegend_")

    # ── Feature bars ─────────────────────────────────────────
    ax.bar(x, z_scores, color=bar_colors, width=0.75,
           edgecolor="none", zorder=2, alpha=0.90)

    # ── Threshold lines ───────────────────────────────────────
    ax.axhline( BASELINE_WARN_SIGMA,  color=FAIL_COLOR, linewidth=1.5,
                linestyle="--", alpha=0.85, zorder=3,
                label=f"Warning  ±{BASELINE_FAIL_SIGMA}σ")
    ax.axhline(-BASELINE_WARN_SIGMA,  color=FAIL_COLOR, linewidth=1.5,
                linestyle="--", alpha=0.85, zorder=3)
    ax.axhline( BASELINE_FAIL_SIGMA,  color=FAIL_COLOR, linewidth=2.0,
                linestyle="--", alpha=0.95, zorder=3,
                label=f"Alert    ±{BASELINE_FAIL_SIGMA}σ")
    ax.axhline(-BASELINE_FAIL_SIGMA,  color=FAIL_COLOR, linewidth=2.0,
                linestyle="--", alpha=0.95, zorder=3)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, alpha=0.6, zorder=1)

    # ── Red 🔴 markers above / below alert bars ───────────────
    for i, (z, c) in enumerate(zip(z_scores, bar_colors)):
        if c == FAIL_COLOR:
            # Place marker just beyond the bar tip
            y_tip    = z + (0.25 if z >= 0 else -0.25)
            ax.text(i, y_tip, "🔴",
                    fontsize=8, ha="center", va="center",
                    zorder=6)

    # ── Feature group dividers + colour labels ────────────────
    groups = [
        (0,  20, "MFCC (×20)",          "#3498DB"),
        (20, 32, "Chroma (×12)",         "#9B59B6"),
        (32, 39, "Spectral\nContrast (×7)", "#1ABC9C"),
        (39, 40, "RMS",                  "#E67E22"),
        (40, 41, "ZCR",                  "#E74C3C"),
    ]
    for (start, end, label, gcolor) in groups:
        mid = (start + end - 1) / 2.0
        # Group label below x-axis
        ax.text(mid, -y_abs_max * 0.97,
                label, color=gcolor, fontsize=8, ha="center",
                va="top", fontweight="bold", zorder=5)
        # Vertical divider between groups
        if start > 0:
            ax.axvline(start - 0.5,
                       color=GRID_COLOR, linewidth=0.8,
                       linestyle=":", alpha=0.6, zorder=1)

    # ── Axes styling ─────────────────────────────────────────
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-y_abs_max, y_abs_max)
    ax.set_xticks(x[::5])
    ax.set_xticklabels([str(i) for i in x[::5]],
                       color=TEXT_COLOR, fontsize=8)
    ax.tick_params(colors=TEXT_COLOR)

    # ── CHANGED: Removed emoji from title to prevent matplotlib render failure ─
    ax.set_title(
        "Anomaly Indicators  |  Per-Feature Deviation from Healthy Baseline",
        color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=10
    )
    ax.set_xlabel("Feature Index  (0–40)",
                  color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Z-Score  (σ deviations from baseline)",
                  color=TEXT_COLOR, fontsize=10)

    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.3, linewidth=0.4)

    ax.legend(
        loc="upper right", framealpha=0.25,
        facecolor=BG_COLOR, edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR, fontsize=9
    )

    # ── Summary badge ─────────────────────────────────────────
    if n_alert >  0:
        badge_color = FAIL_COLOR
        # CHANGED: plain text badge — no emoji in matplotlib text
        badge_text = (f"[ALERT] {n_alert } alerts of {n} features")
    else:
        badge_color = PASS_COLOR
        badge_text  = f"[OK]  All {n} features within healthy range"  # CHANGED: no emoji

    ax.text(
        0.01, 0.97, badge_text,
        transform=ax.transAxes, color=badge_color,
        fontsize=9, va="top", ha="left", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor=BG_COLOR, edgecolor=badge_color, alpha=0.85)
    )

    fig.tight_layout()
    return fig




# ────────────────────────────────────────────────────────────
# 5.  PLOT: WAVEFORM
# ────────────────────────────────────────────────────────────
def plot_waveform(audio, sr, status):
    # CHANGED — WARNING uses orange colour instead of always red/green
    if status == "PASS":
        color = PASS_COLOR
    else:
        color = FAIL_COLOR

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    librosa.display.waveshow(audio, sr=sr, ax=ax, color=color, alpha=0.85)
    ax.set_title("Engine Sound Waveform", color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", color=TEXT_COLOR)
    ax.set_ylabel("Amplitude",      color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID_COLOR, alpha=0.5, linewidth=0.5)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────
# 6.  PLOT: SPECTROGRAM
# ────────────────────────────────────────────────────────────
def plot_spectrogram(audio, sr):
    D = get_spectrogram_data(audio, sr)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    img = librosa.display.specshow(
        D, sr=sr, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="hz", ax=ax, cmap="inferno"
    )
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    ax.set_title("Frequency Spectrogram", color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", color=TEXT_COLOR)
    ax.set_ylabel("Frequency (Hz)", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────
# 7.  PLOT: MEL SPECTROGRAM
# ────────────────────────────────────────────────────────────
def plot_mel_spectrogram(audio, sr):
    mel_db = get_mel_spectrogram_data(audio, sr)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    ax.set_title("Mel Spectrogram", color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", color=TEXT_COLOR)
    ax.set_ylabel("Mel Frequency",  color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────
# 8.  PLOT: CONFIDENCE GAUGE
# ────────────────────────────────────────────────────────────
def plot_confidence_gauge(prob_normal, prob_abnormal, status):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    categories = ["Normal", "Abnormal"]
    values     = [prob_normal * 100, prob_abnormal * 100]
    colors     = [PASS_COLOR, FAIL_COLOR]
    bars = ax.barh(categories, values, color=colors, height=0.5, edgecolor="none")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left",
            color=TEXT_COLOR, fontsize=12, fontweight="bold"
        )
    ax.set_xlim(0, 115)
    ax.set_title("AI Confidence Scores", color=TEXT_COLOR, fontsize=13, fontweight="bold")
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.set_xlabel("Probability (%)", color=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(50, color=GRID_COLOR, linewidth=1, linestyle="--", alpha=0.6)
    ax.grid(axis="x", color=GRID_COLOR, alpha=0.4)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────
# 9.  COMBINED: ALL PLOTS
# ────────────────────────────────────────────────────────────
def generate_all_plots(file_path, status, prob_normal, prob_abnormal):
    """Generate all four core visualisation figures."""
    audio, sr = load_audio(file_path)
    waveform_fig    = plot_waveform(audio, sr, status)
    spectrogram_fig = plot_spectrogram(audio, sr)
    mel_fig         = plot_mel_spectrogram(audio, sr)
    gauge_fig       = plot_confidence_gauge(prob_normal, prob_abnormal, status)
    return waveform_fig, spectrogram_fig, mel_fig, gauge_fig


# ────────────────────────────────────────────────────────────
# 10.  FORMAT RESULT TEXT
# ────────────────────────────────────────────────────────────
def format_result_text(status, confidence, prob_normal, prob_abnormal,
                        pass_threshold, ensemble_explanation=""):
    """
    CHANGED (Task 7) — Updated to show three zones, adaptive threshold,
    and ensemble explanation so the engineer understands every decision.
    """
    # CHANGED — three-zone status icon
    if status == "PASS":
        status_icon = "[PASS]"
    else:
        status_icon = "[FAIL]"

    lines = [
        "=" * 48,
        "  ENGINE DIAGNOSTIC REPORT  (Ensemble System)",
        "=" * 48,
        f"  Final Status     : {status_icon}  ENGINE {status}",
        f"  Confidence       : {confidence:.1f}%",
        "-" * 48,
        f"  RF Normal Score  : {prob_normal   * 100:.1f}%",
        f"  RF Abnorm Score  : {prob_abnormal * 100:.1f}%",
        "-" * 48,
        # CHANGED — show all three thresholds clearly
        f"  PASS  threshold  : >= {pass_threshold*100:.1f}%  (adaptive from training)",
        f"  FAIL  threshold  : <  {pass_threshold*100:.1f}%",
        "-" * 48,
    ]

    if ensemble_explanation:
        lines.append(f"  Ensemble votes   : {ensemble_explanation}")
        lines.append("-" * 48)

    # CHANGED — three-zone result messages
    if status == "PASS":
        lines.append("  Result : No acoustic anomaly detected.")
        lines.append("           Engine sounds within normal range.")
    else:
        lines.append("  Result : ABNORMAL acoustic signature detected!")
        lines.append("           Engine requires immediate inspection.")

    lines.append("=" * 48)
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# 11.  BASELINE COMPARISON (unchanged logic, extracted here)
# ────────────────────────────────────────────────────────────
def compute_baseline_distance(file_path):
    """Compare a test audio file against the healthy baseline."""
    from acoustic_baseline import (
        load_audio_for_baseline, normalize_audio, build_feature_vector
    )
    baseline   = load_baseline()
    mean_vec   = baseline["mean"]
    std_vec    = baseline["std"]
    intra_mean = baseline["intra_mean"]
    intra_std  = baseline["intra_std"]
    n_samples  = baseline["n_samples"]

    audio, sr = load_audio_for_baseline(file_path)
    if audio is None:
        raise RuntimeError("Could not load audio for baseline comparison.")
    audio     = normalize_audio(audio)
    test_fvec = build_feature_vector(audio, sr)

    z_scores = (test_fvec - mean_vec) / std_vec
    distance = float(np.linalg.norm(z_scores))

    sigma = 0.0 if intra_std < 1e-9 else (distance - intra_mean) / intra_std

    if sigma < BASELINE_WARN_SIGMA:
        risk_level = "OK"
    else:
        risk_level = "ALERT"

    report_text = _format_baseline_report(distance, sigma, risk_level,
                                           intra_mean, intra_std, n_samples)
    return distance, sigma, risk_level, report_text


def _format_baseline_report(distance, sigma, risk_level, intra_mean, intra_std, n_samples):
    icon_map = {"OK": "[OK]", "WARNING": "[ALERT]", "ALERT": "[ALERT]"}
    icon     = icon_map.get(risk_level, "?")
    lines = [
        "-" * 45,
        "  BASELINE COMPARISON REPORT",
        "-" * 45,
        f"  Risk Level        : {icon}  {risk_level}",
        f"  Distance Score    : {distance:.4f}",
        f"  Sigma (sigma)     : {sigma:+.2f} std deviations",
        "-" * 45,
        f"  Healthy Baseline  : built from {n_samples} samples",
        f"  Intra-class Mean  : {intra_mean:.4f}",
        f"  Intra-class Std   : {intra_std:.4f}",
        "-" * 45,
    ]
    if risk_level == "OK":
        lines.append("  Interpretation : Sound signature is CLOSE to the baseline.")
    else:
        lines.append("  Interpretation : Sound STRONGLY DEVIATES from baseline.")
        lines.append("                   Immediate inspection advised.")
    lines.append("-" * 45)
    return "\n".join(lines)


def plot_baseline_comparison(distance, sigma, risk_level):
    """Sigma gauge plot."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.barh([0], [BASELINE_WARN_SIGMA], color=PASS_COLOR, height=0.5, alpha=0.35, left=0)
    ax.barh([0], [BASELINE_FAIL_SIGMA - BASELINE_WARN_SIGMA],
            color=FAIL_COLOR, height=0.5, alpha=0.35, left=BASELINE_WARN_SIGMA)
    ax.barh([0], [max(sigma + 2, BASELINE_FAIL_SIGMA + 2) - BASELINE_FAIL_SIGMA],
            color=FAIL_COLOR, height=0.5, alpha=0.25, left=BASELINE_FAIL_SIGMA)
    display_sigma = max(sigma, 0.05)
    color_map     = {"OK": PASS_COLOR, "WARNING": FAIL_COLOR, "ALERT": FAIL_COLOR}
    marker_color  = color_map.get(risk_level, TEXT_COLOR)
    ax.axvline(display_sigma, color=marker_color, linewidth=3, zorder=5)
    ax.text(display_sigma, 0.32, f"sigma={sigma:+.2f}",
            color=marker_color, fontsize=11, fontweight="bold", ha="center", va="bottom")
    ax.text(BASELINE_WARN_SIGMA / 2,                          -0.32, "OK",      color=PASS_COLOR, ha="center", fontsize=9)
    ax.text(BASELINE_FAIL_SIGMA + 1,                          -0.32, "ALERT",   color=FAIL_COLOR, ha="center", fontsize=9)
    ax.axvline(BASELINE_FAIL_SIGMA, color=FAIL_COLOR, linewidth=1, linestyle="--", alpha=0.7)
    max_x = max(sigma + 1.5, BASELINE_FAIL_SIGMA + 2)
    ax.set_xlim(0, max_x)
    ax.set_ylim(-0.5, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Sigma — distance from healthy baseline", color=TEXT_COLOR, fontsize=10)
    ax.set_title("Baseline Distance Gauge", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax.tick_params(colors=TEXT_COLOR)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="x", color=GRID_COLOR, alpha=0.4)
    fig.tight_layout()
    return fig
if __name__ == "__main__":
    file_path = r"C:\Users\Vivek\Downloads\15 sounds\TCL_06_64574928_Atulya_320_Brake_LI_20260326074611.wav"
    
    status, confidence, prediction, prob_normal, prob_abnormal, pass_thr = predict_engine_status(file_path)
    
    print(f"File     : {os.path.basename(file_path)}")
    print(f"Status   : {status}")
    print(f"Normal   : {prob_normal   * 100:.1f}%")
    print(f"Abnormal : {prob_abnormal * 100:.1f}%")
    print(f"Confidence : {confidence:.1f}%")


def get_anomaly_raw_data(file_path):
    """
    Return raw z-score data and feature names for Recharts visualization.
    """
    from acoustic_baseline import (
        load_audio_for_baseline, normalize_audio,
        build_feature_vector, build_feature_names
    )
    try:
        baseline   = load_baseline()
        mean_vec   = baseline["mean"]
        std_vec    = baseline["std"]
        # Use the feature names from the baseline dictionary, fallback to building them
        feat_names = baseline.get("feature_names", build_feature_names())

        audio, sr = load_audio_for_baseline(file_path)
        if audio is None:
            return []
        audio     = normalize_audio(audio)
        test_fvec = build_feature_vector(audio, sr)  # (41,)

        # Z-scores show how many std devs the current engine is from the healthy mean
        z_scores = (test_fvec - mean_vec) / (std_vec + 1e-9)
        
        data = []
        for i, (name, z) in enumerate(zip(feat_names, z_scores)):
            data.append({
                "index": i,
                "name": name,
                "z": float(z),
                "abs_z": float(abs(z))
            })
        return data
    except Exception:
        return []
