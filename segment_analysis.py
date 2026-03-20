# ============================================================
# segment_analysis.py  —  Find the EXACT second where the
#                          engine starts failing
#
# How it works:
#   1. Slice the audio into small overlapping windows
#   2. Extract features from each window
#   3. Run the trained model on each window
#   4. Return the anomaly score for every time position
#   5. Generate a timeline plot showing exactly which seconds failed
# ============================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa

from feature_extraction import (
    load_audio,
    extract_mfcc_features,
    extract_spectral_features,
    extract_chroma_features,
    extract_rms_energy,
    extract_mel_spectrogram_features
)
from train_model import load_model
from config import SAMPLE_RATE, HOP_LENGTH, N_FFT


# ────────────────────────────────────────────────────────────
# COLOURS
# ────────────────────────────────────────────────────────────
PASS_COLOR   = "#2ECC71"
FAIL_COLOR   = "#E74C3C"
WARN_COLOR   = "#F39C12"
BG_COLOR     = "#1A1A2E"
GRID_COLOR   = "#2C2C54"
TEXT_COLOR   = "#ECF0F1"
BAR_NORMAL   = "#1ABC9C"
BAR_ANOMALY  = "#E74C3C"
BAR_WARNING  = "#F39C12"


# ────────────────────────────────────────────────────────────
# 1.  EXTRACT FEATURES FROM ONE SHORT WINDOW OF AUDIO
# ────────────────────────────────────────────────────────────
def extract_window_features(audio_window, sr):
    """
    Extract the same 226-feature vector that train_model.py uses,
    but from a short clip (e.g. 1 second) instead of the full file.

    Parameters
    ----------
    audio_window : np.ndarray   Short audio segment
    sr           : int          Sample rate

    Returns
    -------
    features : np.ndarray  shape (226,)
    """
    # Guard: window must be long enough for FFT
    if len(audio_window) < N_FFT:
        pad_len = N_FFT - len(audio_window)
        audio_window = np.pad(audio_window, (0, pad_len), mode="constant")

    mfcc_mean, mfcc_std = extract_mfcc_features(audio_window, sr)
    spectral             = extract_spectral_features(audio_window, sr)
    chroma               = extract_chroma_features(audio_window, sr)
    rms                  = extract_rms_energy(audio_window)
    mel_mean             = extract_mel_spectrogram_features(audio_window, sr)

    features = np.concatenate([
        mfcc_mean, mfcc_std, spectral, chroma, rms, mel_mean
    ])
    return features


# ────────────────────────────────────────────────────────────
# 2.  SLIDING WINDOW ANALYSIS
# ────────────────────────────────────────────────────────────
def sliding_window_analysis(
    file_path,
    window_sec  = 1.0,     # length of each analysis window in seconds
    hop_sec     = 0.5,     # how far to slide the window each step (overlap)
    threshold   = 0.5      # anomaly score above this → FAIL for that window
):
    """
    Analyse every window of the audio and return a timeline of
    anomaly scores.

    Parameters
    ----------
    file_path   : str
    window_sec  : float   Window length in seconds (default 1.0)
    hop_sec     : float   Step size in seconds     (default 0.5)
                          50% overlap → smooth curve
    threshold   : float   Probability cut-off for FAIL (default 0.5)

    Returns
    -------
    results : list of dict, one entry per window:
        {
          "start_sec"      : float,
          "end_sec"        : float,
          "mid_sec"        : float,   ← centre of window (for plotting)
          "anomaly_score"  : float,   ← 0.0–1.0 (higher = more abnormal)
          "normal_score"   : float,
          "status"         : str,     ← "NORMAL" or "ANOMALY"
          "label"          : str      ← "NORMAL" / "WARNING" / "ANOMALY"
        }

    summary : dict
        {
          "total_windows"   : int,
          "anomaly_windows" : int,
          "worst_start_sec" : float,
          "worst_end_sec"   : float,
          "worst_score"     : float,
          "anomaly_percent" : float,
          "overall_status"  : str     "PASS" or "FAIL"
        }
    """
    model, scaler = load_model()
    audio, sr     = load_audio(file_path)

    window_samples = int(window_sec * sr)
    hop_samples    = int(hop_sec    * sr)
    total_samples  = len(audio)
    total_duration = total_samples / sr

    results = []

    # Slide window across the full recording
    start_sample = 0
    while start_sample + window_samples <= total_samples:
        end_sample   = start_sample + window_samples
        audio_window = audio[start_sample:end_sample]

        start_sec = start_sample / sr
        end_sec   = end_sample   / sr
        mid_sec   = (start_sec + end_sec) / 2.0

        # Extract features and predict
        features        = extract_window_features(audio_window, sr)
        features_scaled = scaler.transform(features.reshape(1, -1))
        probabilities   = model.predict_proba(features_scaled)[0]

        anomaly_score = float(probabilities[1])   # probability of FAIL
        normal_score  = float(probabilities[0])

        # Three-level labelling for better UX
        if anomaly_score >= threshold:
            status = "ANOMALY"
            label  = "ANOMALY"
        elif anomaly_score >= threshold * 0.7:   # 70% of threshold = warning zone
            status = "NORMAL"
            label  = "WARNING"
        else:
            status = "NORMAL"
            label  = "NORMAL"

        results.append({
            "start_sec"    : round(start_sec, 2),
            "end_sec"      : round(end_sec,   2),
            "mid_sec"      : round(mid_sec,   2),
            "anomaly_score": round(anomaly_score, 4),
            "normal_score" : round(normal_score,  4),
            "status"       : status,
            "label"        : label
        })

        start_sample += hop_samples

    # ── Build summary ────────────────────────────────────────
    if not results:
        return [], {"error": "Audio file too short for analysis"}

    anomaly_windows = [r for r in results if r["status"] == "ANOMALY"]
    all_scores      = [r["anomaly_score"] for r in results]
    worst           = max(results, key=lambda r: r["anomaly_score"])

    summary = {
        "total_windows"   : len(results),
        "anomaly_windows" : len(anomaly_windows),
        "worst_start_sec" : worst["start_sec"],
        "worst_end_sec"   : worst["end_sec"],
        "worst_score"     : worst["anomaly_score"],
        "anomaly_percent" : round(len(anomaly_windows) / len(results) * 100, 1),
        "avg_score"       : round(float(np.mean(all_scores)), 4),
        "overall_status"  : "FAIL" if len(anomaly_windows) > 0 else "PASS",
        "total_duration"  : round(total_duration, 2),
        "window_sec"      : window_sec,
        "hop_sec"         : hop_sec,
        "threshold"       : threshold
    }

    return results, summary


# ────────────────────────────────────────────────────────────
# 3.  FIND CONTIGUOUS FAILURE REGIONS
# ────────────────────────────────────────────────────────────
def find_failure_regions(results, gap_tolerance_sec=1.0):
    """
    Merge adjacent ANOMALY windows into continuous failure regions.

    For example, if windows at 3.0s, 3.5s, 4.0s, 4.5s are all
    ANOMALY → one region: 3.0s – 5.0s

    Parameters
    ----------
    results          : list of dicts from sliding_window_analysis()
    gap_tolerance_sec: float   Merge windows that are this close together
                               (prevents splitting one event into many regions)

    Returns
    -------
    regions : list of dict
        {
          "start_sec"  : float,
          "end_sec"    : float,
          "duration"   : float,
          "peak_score" : float,
          "avg_score"  : float,
          "severity"   : str    "HIGH" / "MEDIUM" / "LOW"
        }
    """
    anomaly_windows = [r for r in results if r["status"] == "ANOMALY"]
    if not anomaly_windows:
        return []

    # Sort by time
    anomaly_windows = sorted(anomaly_windows, key=lambda r: r["start_sec"])

    regions = []
    region_start  = anomaly_windows[0]["start_sec"]
    region_end    = anomaly_windows[0]["end_sec"]
    region_scores = [anomaly_windows[0]["anomaly_score"]]

    for window in anomaly_windows[1:]:
        # If this window starts within tolerance of the last region end → extend
        if window["start_sec"] - region_end <= gap_tolerance_sec:
            region_end = max(region_end, window["end_sec"])
            region_scores.append(window["anomaly_score"])
        else:
            # Save completed region
            regions.append(_build_region(region_start, region_end, region_scores))
            # Start new region
            region_start  = window["start_sec"]
            region_end    = window["end_sec"]
            region_scores = [window["anomaly_score"]]

    # Save last region
    regions.append(_build_region(region_start, region_end, region_scores))
    return regions


def _build_region(start_sec, end_sec, scores):
    """Helper: build a region dict from start/end/scores."""
    peak  = max(scores)
    avg   = float(np.mean(scores))
    dur   = round(end_sec - start_sec, 2)
    if peak >= 0.85:
        severity = "HIGH"
    elif peak >= 0.65:
        severity = "MEDIUM"
    else:
        severity = "LOW"
    return {
        "start_sec" : round(start_sec, 2),
        "end_sec"   : round(end_sec,   2),
        "duration"  : dur,
        "peak_score": round(peak, 4),
        "avg_score" : round(avg,  4),
        "severity"  : severity
    }


# ────────────────────────────────────────────────────────────
# 4.  PLOT: ANOMALY TIMELINE (main new plot)
# ────────────────────────────────────────────────────────────
def plot_anomaly_timeline(results, summary, regions, file_path):
    """
    The main new visualisation:
    A bar/filled line chart showing anomaly score at every second,
    with coloured shading, threshold line, and failure regions highlighted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not results:
        return None

    times  = [r["mid_sec"]      for r in results]
    scores = [r["anomaly_score"] for r in results]
    labels = [r["label"]         for r in results]

    threshold = summary["threshold"]

    # Bar colours per window
    bar_colors = []
    for label in labels:
        if label == "ANOMALY":
            bar_colors.append(BAR_ANOMALY)
        elif label == "WARNING":
            bar_colors.append(BAR_WARNING)
        else:
            bar_colors.append(BAR_NORMAL)

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.patch.set_facecolor(BG_COLOR)

    # ── Top plot: anomaly score timeline ────────────────────
    ax = axes[0]
    ax.set_facecolor(BG_COLOR)

    # Shaded failure regions (background highlight first)
    for region in regions:
        ax.axvspan(
            region["start_sec"], region["end_sec"],
            alpha=0.15, color=BAR_ANOMALY,
            label="_nolegend_"
        )

    # Bar chart of anomaly scores
    bar_width = summary["hop_sec"] * 0.85
    ax.bar(times, scores, width=bar_width, color=bar_colors, alpha=0.9, zorder=3)

    # Filled area under the curve for visual continuity
    ax.fill_between(times, scores, alpha=0.15, color=BAR_ANOMALY, zorder=2)

    # Threshold line
    ax.axhline(
        threshold, color=WARN_COLOR, linewidth=1.5,
        linestyle="--", label=f"Threshold ({threshold})", zorder=4
    )

    # Annotate each failure region with a label arrow
    for i, region in enumerate(regions):
        peak_time  = (region["start_sec"] + region["end_sec"]) / 2
        peak_score = region["peak_score"]
        label_text = (
            f"FAULT {i+1}\n"
            f"{region['start_sec']}s – {region['end_sec']}s\n"
            f"Severity: {region['severity']}\n"
            f"Peak: {region['peak_score']:.0%}"
        )
        ax.annotate(
            label_text,
            xy          = (peak_time, peak_score),
            xytext      = (peak_time, min(peak_score + 0.18, 0.95)),
            fontsize    = 8,
            color       = TEXT_COLOR,
            ha          = "center",
            arrowprops  = dict(arrowstyle="->", color=BAR_ANOMALY, lw=1.2),
            bbox        = dict(
                boxstyle = "round,pad=0.3",
                fc       = "#2C1A1A",
                ec       = BAR_ANOMALY,
                alpha    = 0.85
            )
        )

    ax.set_xlim(0, summary["total_duration"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Anomaly Score (0–1)", color=TEXT_COLOR, fontsize=11)
    ax.set_title(
        "Engine Anomaly Timeline — Per-Second Analysis",
        color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=12
    )
    ax.tick_params(colors=TEXT_COLOR)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(True, color=GRID_COLOR, alpha=0.5, linewidth=0.5, zorder=1)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    # Legend
    legend_patches = [
        mpatches.Patch(color=BAR_NORMAL,  label="Normal"),
        mpatches.Patch(color=WARN_COLOR,  label="Warning zone"),
        mpatches.Patch(color=BAR_ANOMALY, label="Anomaly (FAIL)"),
        plt.Line2D([0], [0], color=WARN_COLOR, linestyle="--", label=f"Threshold")
    ]
    ax.legend(
        handles=legend_patches, loc="upper right",
        facecolor="#1A1A2E", edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR, fontsize=9
    )

    # ── Bottom plot: severity colour strip ───────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)
    ax2.set_xlim(0, summary["total_duration"])
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])

    # Draw coloured blocks for each window
    for r in results:
        if r["label"] == "ANOMALY":
            color = BAR_ANOMALY
        elif r["label"] == "WARNING":
            color = WARN_COLOR
        else:
            color = BAR_NORMAL
        ax2.axvspan(r["start_sec"], r["end_sec"], alpha=0.7, color=color)

    # Second markers on the strip
    for sec in range(0, int(summary["total_duration"]) + 1):
        ax2.axvline(sec, color=BG_COLOR, linewidth=0.8)
        ax2.text(
            sec + 0.05, 0.5, f"{sec}s",
            color=TEXT_COLOR, fontsize=7.5, va="center"
        )

    ax2.set_xlabel("Time (seconds)", color=TEXT_COLOR, fontsize=11)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.set_title("Second-by-Second Status Strip", color=TEXT_COLOR, fontsize=10, pad=6)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    fig.tight_layout(h_pad=1.5)
    return fig


# ────────────────────────────────────────────────────────────
# 5.  PLOT: FAILURE REGION DETAIL (zoomed waveform)
# ────────────────────────────────────────────────────────────
def plot_failure_region_zoom(file_path, regions):
    """
    For each failure region, plot a zoomed waveform so engineers
    can see the exact shape of the anomalous sound.

    Returns
    -------
    fig : matplotlib.figure.Figure  or  None if no failures found
    """
    if not regions:
        return None

    audio, sr     = load_audio(file_path)
    n_regions     = len(regions)
    n_rows        = min(n_regions, 3)    # show at most 3 zoomed regions

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), squeeze=False)
    fig.patch.set_facecolor(BG_COLOR)

    for idx in range(n_rows):
        region = regions[idx]
        ax     = axes[idx][0]
        ax.set_facecolor(BG_COLOR)

        # Extract the audio chunk for this region
        # Add 0.5s padding on each side so engineers see context
        pad_sec    = 0.5
        zoom_start = max(0, region["start_sec"] - pad_sec)
        zoom_end   = min(len(audio) / sr, region["end_sec"] + pad_sec)

        start_sample = int(zoom_start * sr)
        end_sample   = int(zoom_end   * sr)
        segment      = audio[start_sample:end_sample]
        t            = np.linspace(zoom_start, zoom_end, len(segment))

        # Split into: pre-fault (gray), fault (red), post-fault (gray)
        fault_mask = (t >= region["start_sec"]) & (t <= region["end_sec"])
        normal_mask = ~fault_mask

        # Plot normal part
        if np.any(normal_mask):
            t_norm = np.where(normal_mask, t, np.nan)
            ax.plot(t_norm, segment, color=BAR_NORMAL, linewidth=0.7, alpha=0.7)

        # Plot fault part
        if np.any(fault_mask):
            t_fault = np.where(fault_mask, t, np.nan)
            ax.plot(t_fault, segment, color=BAR_ANOMALY, linewidth=1.0, alpha=0.9)

        # Highlight fault region background
        ax.axvspan(
            region["start_sec"], region["end_sec"],
            alpha=0.12, color=BAR_ANOMALY
        )
        ax.axvline(region["start_sec"], color=BAR_ANOMALY, linewidth=1.2, linestyle="--", alpha=0.8)
        ax.axvline(region["end_sec"],   color=BAR_ANOMALY, linewidth=1.2, linestyle="--", alpha=0.8)

        # Annotations
        ax.text(
            region["start_sec"], ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] != 0 else 0.5,
            f"  START {region['start_sec']}s",
            color=BAR_ANOMALY, fontsize=8
        )
        ax.text(
            region["end_sec"],
            ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] != 0 else 0.5,
            f"  END {region['end_sec']}s",
            color=BAR_ANOMALY, fontsize=8
        )

        severity_color = {
            "HIGH":   BAR_ANOMALY,
            "MEDIUM": WARN_COLOR,
            "LOW":    "#F0E68C"
        }.get(region["severity"], BAR_ANOMALY)

        ax.set_title(
            f"Fault Region {idx+1} — "
            f"{region['start_sec']}s to {region['end_sec']}s  |  "
            f"Duration: {region['duration']}s  |  "
            f"Severity: {region['severity']}  |  "
            f"Peak Score: {region['peak_score']:.1%}",
            color=severity_color, fontsize=11, fontweight="bold", pad=8
        )
        ax.set_xlabel("Time (seconds)", color=TEXT_COLOR)
        ax.set_ylabel("Amplitude",      color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.4, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

    fig.tight_layout(h_pad=2.0)
    return fig


# ────────────────────────────────────────────────────────────
# 6.  FORMAT TEMPORAL REPORT TEXT
# ────────────────────────────────────────────────────────────
def format_temporal_report(summary, regions):
    """
    Build the detailed text report shown in the Gradio UI.

    Parameters
    ----------
    summary : dict from sliding_window_analysis()
    regions : list from find_failure_regions()

    Returns
    -------
    report : str
    """
    lines = [
        "━" * 52,
        "  TEMPORAL ANOMALY ANALYSIS REPORT",
        "━" * 52,
        f"  Recording duration : {summary['total_duration']}s",
        f"  Windows analysed   : {summary['total_windows']}",
        f"  Window size        : {summary['window_sec']}s  |  Step: {summary['hop_sec']}s",
        "─" * 52,
        f"  Anomaly windows    : {summary['anomaly_windows']} / {summary['total_windows']}",
        f"  Anomalous time     : {summary['anomaly_percent']}% of recording",
        f"  Average score      : {summary['avg_score']:.1%}",
        f"  Worst score        : {summary['worst_score']:.1%}  at {summary['worst_start_sec']}s",
        "─" * 52,
    ]

    if not regions:
        lines.append("  ✅  No failure regions detected.")
        lines.append("       Engine sounds normal throughout recording.")
    else:
        lines.append(f"  ❌  {len(regions)} Failure Region(s) Detected:")
        lines.append("")
        for i, region in enumerate(regions, start=1):
            severity_icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}.get(region["severity"], "🔴")
            lines += [
                f"  Fault #{i}  {severity_icon} {region['severity']} severity",
                f"    Start     : {region['start_sec']}s",
                f"    End       : {region['end_sec']}s",
                f"    Duration  : {region['duration']}s",
                f"    Peak score: {region['peak_score']:.1%}",
                f"    Avg score : {region['avg_score']:.1%}",
                ""
            ]

    lines += ["━" * 52]

    if regions:
        lines += [
            "  RECOMMENDATION:",
            "  Inspect engine at the highlighted time region(s).",
            "  High-severity faults require immediate attention.",
            "━" * 52
        ]
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# 7.  MASTER FUNCTION: run everything and return all outputs
# ────────────────────────────────────────────────────────────
def run_temporal_analysis(
    file_path,
    window_sec = 1.0,
    hop_sec    = 0.5,
    threshold  = 0.5
):
    """
    Run the full temporal analysis pipeline for one audio file.

    Returns
    -------
    report_text  : str
    timeline_fig : matplotlib Figure
    zoom_fig     : matplotlib Figure  (or None if no faults)
    summary      : dict
    regions      : list of dict
    """
    # Step 1: sliding window
    results, summary = sliding_window_analysis(
        file_path, window_sec, hop_sec, threshold
    )

    if "error" in summary:
        return summary["error"], None, None, summary, []

    # Step 2: find contiguous regions
    regions = find_failure_regions(results)

    # Step 3: plots
    timeline_fig = plot_anomaly_timeline(results, summary, regions, file_path)
    zoom_fig     = plot_failure_region_zoom(file_path, regions)

    # Step 4: text report
    report_text = format_temporal_report(summary, regions)

    return report_text, timeline_fig, zoom_fig, summary, regions
