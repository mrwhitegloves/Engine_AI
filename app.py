# ============================================================
# app.py  —  Gradio web interface for the Engine Diagnostic AI
#
# TABS:
#   Tab 1 — Engine Diagnosis      (RF classifier + baseline comparison)
#   Tab 2 — Baseline Builder      (build / rebuild baseline from audio)
#   Tab 3 — Multi-File Analysis   (NEW — batch WAV stats, FFT, spectrogram,
#                                         group comparison, anomaly detection)
#
# Run:   python app.py
# Open:  http://localhost:7860
# ============================================================

import os
import wave
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from scipy import stats as scipy_stats

from predict import (
    predict_engine_status,
    generate_all_plots,
    format_result_text,
    compute_baseline_distance,
    plot_baseline_comparison,
    plot_anomaly_indicators,
    compute_ensemble_status,
    compute_weighted_zscore_vote,
    load_feature_importances,
    get_anomaly_raw_data,
)
from config import BASELINE_PATH, BASELINE_PLOT, NORMAL_DIR

# ── shared plot style ─────────────────────────────────────────
BG   = "#1A1A2E"
TEXT = "#ECF0F1"
GRID = "#2C2C54"


# ════════════════════════════════════════════════════════════
# STATUS CHECKERS
# ════════════════════════════════════════════════════════════

def check_model_ready():
    return (os.path.exists("models/engine_model.pkl") and
            os.path.exists("models/scaler.pkl"))

def check_baseline_ready():
    return os.path.exists(BASELINE_PATH)

def get_baseline_status_text():
    if check_baseline_ready():
        import joblib
        try:
            data = joblib.load(BASELINE_PATH)
            n    = data.get("n_samples", "?")
            return f"Baseline ready — built from {n} healthy sample(s)"
        except Exception:
            return "Baseline file exists (could not read details)"
    return "No baseline built yet — use the Baseline Builder tab"


# ════════════════════════════════════════════════════════════
# TAB 1 FUNCTIONS  —  Engine Diagnosis (unchanged)
# ════════════════════════════════════════════════════════════

def analyze_engine_audio(audio_file):
    """Single-file engine diagnosis using RF + ensemble."""
    if audio_file is None:
        return ("Please upload an engine audio file first.",
                None, None, None, None,
                "Upload a file to see baseline comparison.",
                None, None)

    if not check_model_ready():
        msg = ("RF Model not found!\n\nPlease train first:\n"
               "   python train_model.py\n\nThen restart this app.")
        return msg, None, None, None, None, "RF model not ready.", None, None

    try:
        status, confidence, prediction, prob_normal, prob_abnormal, pass_threshold = \
            predict_engine_status(audio_file)
    except Exception as err:
        return (f"Classifier error:\n{err}",
                None, None, None, None, "Classifier failed.", None, None)

    risk_level     = "N/A"
    baseline_text  = ""
    baseline_gauge = None

    if check_baseline_ready():
        try:
            distance, sigma, risk_level, baseline_text = \
                compute_baseline_distance(audio_file)
            baseline_gauge = plot_baseline_comparison(distance, sigma, risk_level)
        except Exception as err:
            baseline_text = f"Baseline comparison error:\n{err}"
            risk_level    = "N/A"
    else:
        baseline_text = ("No baseline built yet.\n\n"
                         "Go to the Baseline Builder tab and click\n"
                         "Build Baseline to enable this analysis.")

    try:
        importances = load_feature_importances()
        zscore_vote, _ = compute_weighted_zscore_vote(audio_file, importances)
    except Exception:
        zscore_vote = "PASS"

    try:
        final_status, ensemble_explanation = compute_ensemble_status(
            prob_normal, risk_level, zscore_vote)
    except Exception:
        final_status         = status
        ensemble_explanation = "(ensemble error, using RF only)"

    print("final_status", final_status)
    print("ensemble_explanation", ensemble_explanation)

    rf_text = format_result_text(final_status, confidence,
                                  prob_normal, prob_abnormal,
                                  pass_threshold, ensemble_explanation)

    try:
        waveform_fig, spectrogram_fig, mel_fig, gauge_fig = \
            generate_all_plots(audio_file, final_status, prob_normal, prob_abnormal)
    except Exception:
        waveform_fig = spectrogram_fig = mel_fig = gauge_fig = None

    anomaly_fig = None
    if check_baseline_ready():
        try:
            anomaly_fig = plot_anomaly_indicators(audio_file)
        except Exception:
            anomaly_fig = None

    return (rf_text, waveform_fig, spectrogram_fig, mel_fig, gauge_fig,
            baseline_text, baseline_gauge, anomaly_fig,
            prob_normal, prob_abnormal, float(pass_threshold),
            distance if 'distance' in locals() else 0.0,
            sigma if 'sigma' in locals() else 0.0,
            get_anomaly_raw_data(audio_file))


# ════════════════════════════════════════════════════════════
# TAB 2 FUNCTIONS  —  Baseline Builder (unchanged)
# ════════════════════════════════════════════════════════════

def run_baseline_build():
    logs = []
    def log(msg): logs.append(str(msg))
    yield "\n".join(logs), "Building baseline ...", None
    try:
        from acoustic_baseline import build_acoustic_baseline
        build_acoustic_baseline(audio_folder=NORMAL_DIR, log_fn=log)
        yield ("\n".join(logs), get_baseline_status_text(),
               BASELINE_PLOT if os.path.exists(BASELINE_PLOT) else None)
    except FileNotFoundError as err:
        logs.append(f"\n{err}")
        yield "\n".join(logs), "Build failed — see log", None
    except Exception as err:
        logs.append(f"\nUnexpected error: {err}")
        yield "\n".join(logs), "Build failed — see log", None

def reload_baseline_status():
    img = BASELINE_PLOT if os.path.exists(BASELINE_PLOT) else None
    return get_baseline_status_text(), img


# ════════════════════════════════════════════════════════════
# TAB 3 BACKEND — Multi-File Audio Analysis  (ALL NEW)
# ════════════════════════════════════════════════════════════

# ── Keyword list that marks a file as "Bad Signal" ────────────
BAD_KEYWORDS = ["abnormal", "damper", "defective", "cam", "valvelash",
                "bad", "fault", "fail", "noise", "knock"]


def _classify_file(filename):
    """Return 'Bad Signals' or 'Good Signals' based on filename keywords."""
    fname_lower = filename.lower()
    return "Bad Signals" if any(k in fname_lower for k in BAD_KEYWORDS) else "Good Signals"


def _load_wav_bytes(file_obj):
    """
    Load a WAV file from a Gradio file object.
    Gradio gives us a file path string (temp file), not raw bytes.
    Returns (signal_array, framerate, n_frames, n_channels) or None on error.
    """
    try:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        with wave.open(path, "rb") as wf:
            n_frames   = wf.getnframes()
            framerate  = wf.getframerate()
            n_channels = wf.getnchannels()
            raw        = wf.readframes(n_frames)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if n_channels == 2:
            arr = arr.reshape(-1, 2).mean(axis=1)
        return arr, framerate, n_frames, n_channels
    except Exception:
        return None


def _compute_stats_for_file(filename, arr, framerate, n_frames, n_channels):
    """Compute the full statistics row for one audio file (mirrors the Streamlit code)."""
    fft_vals  = np.abs(fft(arr))
    freqs     = fftfreq(len(arr), 1.0 / framerate)[: len(arr) // 2]
    peak_freq = float(freqs[np.argmax(fft_vals[: len(arr) // 2])])
    rms       = float(np.sqrt(np.mean(arr ** 2)))

    return {
        "Filename"          : filename,
        "Group"             : _classify_file(filename),
        "Duration_s"        : round(n_frames / framerate, 3),
        "Sample_Rate"       : framerate,
        "Channels"          : n_channels,
        "Mean"              : round(float(np.mean(arr)), 3),
        "Median"            : round(float(np.median(arr)), 3),
        "Std_Dev"           : round(float(np.std(arr)), 3),
        "Min"               : round(float(np.min(arr)), 1),
        "Max"               : round(float(np.max(arr)), 1),
        "Range"             : round(float(np.max(arr) - np.min(arr)), 1),
        "RMS"               : round(rms, 3),
        "Peak_Amplitude"    : round(float(np.max(np.abs(arr))), 1),
        "Crest_Factor"      : round(float(np.max(np.abs(arr))) / (rms + 1e-9), 3),
        "Peak_Frequency_Hz" : round(peak_freq, 2),
        "Energy"            : round(float(np.sum(arr ** 2)), 1),
        "Zero_Crossings"    : int(np.sum(np.abs(np.diff(np.sign(arr)))) // 2),
    }


# ── Colours shared across all Tab-3 plots ─────────────────────
_GROUP_COLORS = {"Bad Signals": "#E74C3C", "Good Signals": "#2ECC71"}

def _style_ax(ax, title, xlabel, ylabel):
    """Apply dark-theme styling to a matplotlib axes."""
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, alpha=0.4, linewidth=0.5)


# ── 1. PROCESS uploaded files → stats DataFrame + waveform dict ──

def process_multi_files(files):
    """
    Called when the user clicks 'Analyse All Files'.
    Accepts a list of Gradio file paths.
    Returns: (stats_df, waveform_dict, status_message)
    """
    if not files:
        return None, {}, "No files uploaded."

    stats_list    = []
    waveform_dict = {}   # filename -> {arr, framerate}

    for file_obj in files:
        # Gradio passes the temp path as a string
        path     = file_obj if isinstance(file_obj, str) else file_obj.name
        filename = os.path.basename(path)
        result   = _load_wav_bytes(path)
        if result is None:
            continue
        arr, framerate, n_frames, n_channels = result
        stats_list.append(_compute_stats_for_file(
            filename, arr, framerate, n_frames, n_channels))
        waveform_dict[filename] = {"arr": arr, "framerate": framerate}

    if not stats_list:
        return None, {}, "Could not load any WAV files."

    df = pd.DataFrame(stats_list)
    msg = f"Loaded {len(df)} file(s): {df['Group'].value_counts().to_dict()}"
    return df, waveform_dict, msg


# ── 2. OVERVIEW statistics bar charts ─────────────────────────

def make_overview_charts(df):
    """Return 4 bar-chart figures: RMS, Duration, Peak Freq, Zero Crossings."""
    metrics = [
        ("RMS",               "RMS Amplitude by File"),
        ("Duration_s",        "Duration (s) by File"),
        ("Peak_Frequency_Hz", "Peak Frequency (Hz) by File"),
        ("Zero_Crossings",    "Zero Crossings by File"),
    ]
    figs = []
    for col, title in metrics:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor(BG)
        colors = [_GROUP_COLORS.get(g, "#3498DB") for g in df["Group"]]
        bars = ax.bar(range(len(df)), df[col], color=colors, edgecolor="none")
        # Short filename labels on x-axis
        short_names = [n[:18] + ".." if len(n) > 20 else n for n in df["Filename"]]
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        _style_ax(ax, title, "", col)
        # Legend patch
        from matplotlib.patches import Patch
        legend_patches = [Patch(color=c, label=g) for g, c in _GROUP_COLORS.items()
                          if g in df["Group"].values]
        if legend_patches:
            ax.legend(handles=legend_patches, loc="upper right",
                      facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
        fig.tight_layout()
        figs.append(fig)
        plt.close(fig)
    return figs   # list of 4 figures


# ── 3. WAVEFORM for selected file ──────────────────────────────

def make_waveform_plot(df, waveform_dict, selected_filename):
    """Return a waveform figure for the selected filename."""
    if df is None or selected_filename not in waveform_dict:
        return None
    entry    = waveform_dict[selected_filename]
    arr      = entry["arr"]
    fr       = entry["framerate"]
    group    = df.loc[df["Filename"] == selected_filename, "Group"].iloc[0]
    color    = _GROUP_COLORS.get(group, "#3498DB")
    time_arr = np.linspace(0, len(arr) / fr, num=len(arr))

    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor(BG)
    ax.plot(time_arr, arr, color=color, linewidth=0.6, alpha=0.9)
    _style_ax(ax, f"Waveform — {selected_filename}", "Time (s)", "Amplitude")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ── 4. FFT for selected file ───────────────────────────────────

def make_fft_plot(df, waveform_dict, selected_filename, freq_limit=5000):
    """Return an FFT magnitude figure for the selected filename."""
    if df is None or selected_filename not in waveform_dict:
        return None
    entry = waveform_dict[selected_filename]
    arr   = entry["arr"]
    fr    = entry["framerate"]
    group = df.loc[df["Filename"] == selected_filename, "Group"].iloc[0]
    color = _GROUP_COLORS.get(group, "#E74C3C")

    fft_vals = np.abs(fft(arr))
    freqs    = fftfreq(len(arr), 1.0 / fr)[: len(arr) // 2]
    mask     = freqs <= freq_limit

    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor(BG)
    ax.plot(freqs[mask], fft_vals[: len(arr) // 2][mask],
            color=color, linewidth=0.8, alpha=0.9)
    ax.fill_between(freqs[mask], fft_vals[: len(arr) // 2][mask],
                    color=color, alpha=0.15)
    _style_ax(ax, f"FFT Spectrum — {selected_filename}",
              "Frequency (Hz)", "Magnitude")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ── 5. SPECTROGRAM for selected file ──────────────────────────

def make_spectrogram_plot(df, waveform_dict, selected_filename):
    """Return a spectrogram (dB heatmap) figure for the selected filename."""
    if df is None or selected_filename not in waveform_dict:
        return None
    entry = waveform_dict[selected_filename]
    arr   = entry["arr"]
    fr    = entry["framerate"]

    freqs, times, Sxx = scipy_signal.spectrogram(
        arr.astype(np.float64), fs=fr, nperseg=1024)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    img = ax.pcolormesh(times, freqs, Sxx_db, cmap="viridis", shading="auto")
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
    _style_ax(ax, f"Spectrogram — {selected_filename}",
              "Time (s)", "Frequency (Hz)")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ── 6. GROUP COMPARISON charts ─────────────────────────────────

def make_group_comparison_charts(df):
    """
    Return:
      pie_fig         — group distribution pie chart
      boxplot_figs    — list of 6 box-plot figures (one per metric)
    """
    if df is None:
        return None, []

    # ── Pie chart ─────────────────────────────────────────────
    group_counts = df["Group"].value_counts()
    fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
    fig_pie.patch.set_facecolor(BG)
    ax_pie.set_facecolor(BG)
    colors_pie = [_GROUP_COLORS.get(g, "#3498DB") for g in group_counts.index]
    wedges, texts, autotexts = ax_pie.pie(
        group_counts.values, labels=group_counts.index,
        colors=colors_pie, autopct="%1.0f%%",
        textprops={"color": TEXT}, startangle=90)
    for at in autotexts:
        at.set_color(BG)
        at.set_fontweight("bold")
    ax_pie.set_title("Signal Group Distribution", color=TEXT, fontsize=12,
                     fontweight="bold")
    fig_pie.tight_layout()
    plt.close(fig_pie)

    # ── Box plots ──────────────────────────────────────────────
    metrics   = ["RMS", "Peak_Frequency_Hz", "Zero_Crossings",
                 "Energy", "Crest_Factor", "Duration_s"]
    box_figs  = []
    groups    = df["Group"].unique()
    positions = list(range(len(groups)))

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(BG)
        data_per_group = [df[df["Group"] == g][metric].dropna().values for g in groups]
        bp = ax.boxplot(data_per_group, positions=positions, patch_artist=True,
                        widths=0.5,
                        boxprops   =dict(linewidth=1.5),
                        medianprops=dict(color=TEXT, linewidth=2),
                        whiskerprops=dict(color=TEXT),
                        capprops   =dict(color=TEXT),
                        flierprops =dict(marker="o", markersize=4))
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(_GROUP_COLORS.get(g, "#3498DB"))
            patch.set_alpha(0.75)
        ax.set_xticks(positions)
        ax.set_xticklabels(groups, color=TEXT, fontsize=9)
        _style_ax(ax, f"{metric} Distribution", "Group", metric)
        fig.tight_layout()
        plt.close(fig)
        box_figs.append(fig)

    return fig_pie, box_figs


# ── 7. ANOMALY DETECTION ───────────────────────────────────────

def run_anomaly_detection(df):
    """
    Compute anomaly thresholds from Good Signals (mean ± 2·std).
    Returns:
      anomaly_df     — per-file anomaly summary DataFrame
      scatter_fig    — RMS vs Peak Frequency scatter (X = anomalous)
      t_test_df      — T-test significance table
      metrics_ok     — list of significant metric names
    """
    if df is None or "Good Signals" not in df["Group"].values:
        return None, None, None, []

    metrics = ["RMS", "Peak_Frequency_Hz", "Zero_Crossings",
               "Energy", "Crest_Factor", "Duration_s"]

    good = df[df["Group"] == "Good Signals"]

    # ── Thresholds from good signals ──────────────────────────
    thresholds = {}
    for m in metrics:
        mu  = good[m].mean()
        std = good[m].std()
        thresholds[m] = {"lower": mu - 2 * std,
                         "upper": mu + 2 * std,
                         "mean" : mu, "std": std}

    # ── Per-file anomaly count ─────────────────────────────────
    rows = []
    for _, row in df.iterrows():
        n_anom   = sum(
            1 for m in metrics
            if row[m] < thresholds[m]["lower"] or row[m] > thresholds[m]["upper"]
        )
        rows.append({
            "Filename"     : row["Filename"],
            "Group"        : row["Group"],
            "Anomaly_Count": n_anom,
            "Status"       : "Anomalous" if n_anom > 0 else "Normal",
        })
    anomaly_df = pd.DataFrame(rows)

    # ── Scatter plot: RMS vs Peak Frequency ───────────────────
    fig_sc, ax_sc = plt.subplots(figsize=(10, 5))
    fig_sc.patch.set_facecolor(BG)
    ax_sc.set_facecolor(BG)

    for _, row in df.iterrows():
        fname  = row["Filename"]
        group  = row["Group"]
        color  = _GROUP_COLORS.get(group, "#3498DB")
        is_bad = anomaly_df.loc[
            anomaly_df["Filename"] == fname, "Anomaly_Count"].iloc[0] > 0
        marker = "X" if is_bad else "o"
        size   = 120 if is_bad else 80
        ax_sc.scatter(row["RMS"], row["Peak_Frequency_Hz"],
                      c=color, marker=marker, s=size, zorder=3,
                      linewidths=1.5,
                      edgecolors="white" if is_bad else "none",
                      label=f"{'[!] ' if is_bad else ''}{fname[:20]}")

    _style_ax(ax_sc,
              "Anomaly Detection: RMS vs Peak Frequency\n"
              "(X marks = anomalous files, circles = normal)",
              "RMS Amplitude", "Peak Frequency (Hz)")
    fig_sc.tight_layout()
    plt.close(fig_sc)

    # ── T-test significance table ──────────────────────────────
    t_rows = []
    if "Bad Signals" in df["Group"].values:
        bad = df[df["Group"] == "Bad Signals"]
        for m in metrics:
            bv = bad[m].dropna()
            gv = good[m].dropna()
            if len(bv) >= 2 and len(gv) >= 2:
                t, p = scipy_stats.ttest_ind(bv, gv, equal_var=False)
                t_rows.append({
                    "Metric"     : m,
                    "T-Statistic": round(t, 3),
                    "P-Value"    : round(p, 4),
                    "Significant": "Yes" if p < 0.05 else "No",
                    "Bad_Mean"   : round(bv.mean(), 3),
                    "Good_Mean"  : round(gv.mean(), 3),
                    "Difference" : round(bv.mean() - gv.mean(), 3),
                })
    t_test_df = pd.DataFrame(t_rows) if t_rows else None
    sig_metrics = ([r["Metric"] for r in t_rows if r["Significant"] == "Yes"]
                   if t_rows else [])

    return anomaly_df, fig_sc, t_test_df, sig_metrics


# ── 8. TOP-LEVEL FUNCTION wired to the Analyse button ─────────

def analyse_all_files(files):
    """
    Master function called when user clicks 'Analyse All Files' in Tab 3.
    Runs every computation and returns all outputs in one shot.

    Returns (40 values — must match outputs list in build_ui):
      status_msg           str
      stats_df             DataFrame
      rms_fig              Figure
      dur_fig              Figure
      peakfreq_fig         Figure
      zcr_fig              Figure
      filenames_choices    list  (for dropdown updates)
      pie_fig              Figure
      box_rms_fig          Figure
      box_pf_fig           Figure
      box_zcr_fig          Figure
      box_energy_fig       Figure
      box_cf_fig           Figure
      box_dur_fig          Figure
      anomaly_df           DataFrame
      scatter_fig          Figure
      t_test_df            DataFrame  or  None
      sig_msg              str
    """
    # ── Process files ──────────────────────────────────────────
    df, waveform_dict, status_msg = process_multi_files(files)

    if df is None:
        empty = [None] * 19
        empty[0] = status_msg
        return tuple(empty)

    # Store waveform_dict in a temporary module-level cache so the
    # per-file view functions can read it without re-processing.
    _WAVEFORM_CACHE.clear()
    _WAVEFORM_CACHE.update(waveform_dict)
    _DF_CACHE.clear()
    _DF_CACHE["df"] = df

    # ── Overview bar charts ────────────────────────────────────
    bar_figs = make_overview_charts(df)  # 4 figures

    # ── Filenames list for dropdown ────────────────────────────
    filenames = df["Filename"].tolist()

    # ── Group comparison ───────────────────────────────────────
    pie_fig, box_figs = make_group_comparison_charts(df)

    # ── Anomaly detection ──────────────────────────────────────
    anomaly_df, scatter_fig, t_test_df, sig_metrics = run_anomaly_detection(df)

    sig_msg = ""
    if sig_metrics:
        sig_msg = "Significant differences found in: " + ", ".join(sig_metrics)
    elif t_test_df is not None:
        sig_msg = "No statistically significant differences detected between groups."
    else:
        sig_msg = "T-test skipped (need both Good and Bad signal files)."

    # NEW — build metadata table for Section 5
    metadata_table = make_metadata_table(df)                   # NEW

    return (
        status_msg,
        df,
        bar_figs[0],  # RMS chart
        bar_figs[1],  # Duration chart
        bar_figs[2],  # Peak Freq chart
        bar_figs[3],  # ZCR chart
        filenames,
        pie_fig,
        box_figs[0] if len(box_figs) > 0 else None,
        box_figs[1] if len(box_figs) > 1 else None,
        box_figs[2] if len(box_figs) > 2 else None,
        box_figs[3] if len(box_figs) > 3 else None,
        box_figs[4] if len(box_figs) > 4 else None,
        box_figs[5] if len(box_figs) > 5 else None,
        anomaly_df,
        scatter_fig,
        t_test_df if t_test_df is not None else pd.DataFrame(),
        sig_msg,
        metadata_table,    # NEW — 18th output
    )


# ── Simple in-memory caches (dict shared across callbacks) ────
_WAVEFORM_CACHE = {}
_DF_CACHE       = {}


def view_waveform(selected_filename):
    """Called when user picks a different file in the waveform dropdown."""
    df = _DF_CACHE.get("df")
    return make_waveform_plot(df, _WAVEFORM_CACHE, selected_filename)


def view_fft(selected_filename, freq_limit):
    """Called when user picks file or adjusts the freq-limit slider."""
    df = _DF_CACHE.get("df")
    return make_fft_plot(df, _WAVEFORM_CACHE, selected_filename,
                         int(freq_limit))


def view_spectrogram(selected_filename):
    """Called when user picks a different file in the spectrogram dropdown."""
    df = _DF_CACHE.get("df")
    return make_spectrogram_plot(df, _WAVEFORM_CACHE, selected_filename)


# ── NEW 1: Build baseline from uploaded files (Tab 2 upload option) ──────────
def build_baseline_from_uploads(uploaded_files, log_box):
    """
    NEW — Allows the user to upload .wav files directly in Tab 2
    instead of putting them in dataset/normal/ manually.

    Steps:
      1. Save each uploaded file into dataset/normal/
      2. Call the existing build_acoustic_baseline() pipeline
    Yields log lines back to the UI.
    """
    import shutil
    from acoustic_baseline import build_acoustic_baseline

    logs = []
    def log(msg):
        logs.append(str(msg))

    yield "\n".join(logs), "Saving uploaded files ...", None

    if not uploaded_files:
        yield "No files uploaded.", "Upload .wav files first.", None
        return

    # Save uploaded files into dataset/normal/
    os.makedirs(NORMAL_DIR, exist_ok=True)
    saved = 0
    for fobj in uploaded_files:
        src_path = fobj if isinstance(fobj, str) else fobj.name
        fname    = os.path.basename(src_path)
        dest     = os.path.join(NORMAL_DIR, fname)
        try:
            shutil.copy2(src_path, dest)
            logs.append(f"  Saved: {fname}")
            saved += 1
        except Exception as e:
            logs.append(f"  SKIP {fname}: {e}")

    logs.append(f"\n  {saved} file(s) saved to {NORMAL_DIR}")
    yield "\n".join(logs), f"Saved {saved} file(s) — building baseline ...", None

    try:
        build_acoustic_baseline(audio_folder=NORMAL_DIR, log_fn=log)
        yield ("\n".join(logs), get_baseline_status_text(),
               BASELINE_PLOT if os.path.exists(BASELINE_PLOT) else None)
    except Exception as err:
        logs.append(f"\nError: {err}")
        yield "\n".join(logs), "Build failed — see log", None


# ── NEW 2: Load files from a folder path (Tab 3 folder input option) ─────────
def load_files_from_folder(folder_path):
    """
    NEW — Alternative to file upload: user types a folder path,
    all .wav files in that folder are loaded and analysed.
    Returns same tuple as analyse_all_files().
    """
    from pathlib import Path

    if not folder_path or not folder_path.strip():
        empty = [None] * 19
        empty[0] = "Enter a folder path and click Load Folder."
        return tuple(empty)

    folder_path = folder_path.strip()
    if not os.path.isdir(folder_path):
        empty = [None] * 19
        empty[0] = f"Folder not found: {folder_path}"
        return tuple(empty)

    wav_files = list(Path(folder_path).glob("*.wav"))
    if not wav_files:
        empty = [None] * 19
        empty[0] = f"No .wav files found in: {folder_path}"
        return tuple(empty)

    # Build fake file-object list (just paths as strings)
    return analyse_all_files([str(p) for p in sorted(wav_files)])


# ── NEW 3: Metadata analysis (parse underscore-separated filename parts) ──────
def make_metadata_table(df):
    """
    NEW — Parse every filename into underscore-separated parts and return
    a DataFrame showing Part_1, Part_2, ... for each file.
    Mirrors Streamlit Tab 5 (Metadata Analysis).
    """
    if df is None:
        return pd.DataFrame()

    rows = []
    for fname in df["Filename"]:
        base  = os.path.splitext(fname)[0]
        parts = base.split("_")
        row   = {"Filename": fname}
        for i, p in enumerate(parts, start=1):
            row[f"Part_{i}"] = p
        rows.append(row)
    return pd.DataFrame(rows)


# ── NEW 4: Zoomed waveform plot ───────────────────────────────────────────────
def view_waveform_zoomed(selected_filename, start_s, end_s):
    """
    NEW — Waveform zoomed to [start_s, end_s] seconds.
    Mirrors the Streamlit waveform zoom slider feature.
    """
    df = _DF_CACHE.get("df")
    if df is None or selected_filename not in _WAVEFORM_CACHE:
        return None

    entry    = _WAVEFORM_CACHE[selected_filename]
    arr      = entry["arr"]
    fr       = entry["framerate"]
    group    = df.loc[df["Filename"] == selected_filename, "Group"].iloc[0]
    color    = _GROUP_COLORS.get(group, "#3498DB")
    time_arr = np.linspace(0, len(arr) / fr, num=len(arr))

    # Clamp the range
    total_s  = float(time_arr[-1])
    start_s  = float(np.clip(start_s, 0, total_s))
    end_s    = float(np.clip(end_s,   0, total_s))
    if end_s <= start_s:
        end_s = min(start_s + 0.5, total_s)

    mask     = (time_arr >= start_s) & (time_arr <= end_s)
    t_zoom   = time_arr[mask]
    a_zoom   = arr[mask]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor(BG)
    ax.plot(t_zoom, a_zoom, color=color, linewidth=0.8, alpha=0.95)
    _style_ax(ax,
              f"Zoomed Waveform — {selected_filename}  [{start_s:.2f}s – {end_s:.2f}s]",
              "Time (s)", "Amplitude")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ── NEW 5: Anomaly metrics summary (total / anomalous / rate) ─────────────────
def get_anomaly_metrics(anomaly_df):
    """
    NEW — Returns (total_files, anomalous_files, anomaly_rate_pct) strings
    for the three Metric widgets in Tab 3, mirroring Streamlit's metric boxes.
    """
    if anomaly_df is None or len(anomaly_df) == 0:
        return "—", "—", "—"
    total     = len(anomaly_df)
    anomalous = int((anomaly_df["Status"] == "Anomalous").sum())
    rate      = f"{anomalous / total * 100:.1f}%"
    return str(total), str(anomalous), rate


# ── NEW 6: Per-file anomaly detail text ───────────────────────────────────────
def make_anomaly_detail_text(files_state):
    """
    NEW — Returns a formatted string showing exactly which metrics are
    anomalous for each flagged file, mirroring Streamlit's expander blocks.
    """
    df = _DF_CACHE.get("df")
    if df is None:
        return "No data loaded."

    metrics = ["RMS", "Peak_Frequency_Hz", "Zero_Crossings",
               "Energy", "Crest_Factor", "Duration_s"]

    good = df[df["Group"] == "Good Signals"]
    if len(good) == 0:
        return "No Good Signal files available to compute thresholds."

    thresholds = {}
    for m in metrics:
        mu  = good[m].mean()
        std = good[m].std()
        thresholds[m] = {"lower": mu - 2*std, "upper": mu + 2*std,
                         "mean": mu, "std": std}

    lines = []
    for _, row in df.iterrows():
        anomalies = []
        for m in metrics:
            lo = thresholds[m]["lower"]
            hi = thresholds[m]["upper"]
            v  = row[m]
            if v < lo or v > hi:
                dev = abs(v - thresholds[m]["mean"]) / (thresholds[m]["std"] + 1e-9)
                anomalies.append(
                    f"    {m}: {v:.3f}  "
                    f"(normal range [{lo:.3f}, {hi:.3f}], "
                    f"deviation = {dev:.2f} std)"
                )
        if anomalies:
            lines.append(f"[ANOMALOUS] {row['Filename']}  ({row['Group']})")
            lines.extend(anomalies)
            lines.append("")

    return "\n".join(lines) if lines else "No anomalous files detected."



# ════════════════════════════════════════════════════════════
# FLASK REST API
# ════════════════════════════════════════════════════════════

import base64
import json
import io
import tempfile
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

# Point Flask to the React build folder
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")

flask_app = Flask(
    __name__, 
    static_folder=FRONTEND_DIST,
    static_url_path="",
    template_folder=FRONTEND_DIST
)
CORS(flask_app)


# ── Global error handlers (always return JSON, never HTML) ────

@flask_app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found", "detail": str(e)}), 404

@flask_app.errorhandler(500)
def internal_error(e):
    import traceback
    traceback.print_exc()
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500

@flask_app.errorhandler(Exception)
def unhandled_exception(e):
    import traceback
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


# ── Serialisation helpers ─────────────────────────────────────

def fig_to_b64(fig):
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    if fig is None:
        return None
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    except Exception:
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def df_to_records(df):
    """Convert a DataFrame to a JSON-safe list of record dicts."""
    if df is None:
        return []
    if hasattr(df, "empty") and df.empty:
        return []
    return json.loads(df.where(pd.notnull(df), None).to_json(orient="records"))


def img_path_to_b64(path):
    """Read an image file from disk and return base64 string."""
    if not path or not os.path.exists(str(path)):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_upload(file_obj):
    """Save a Flask uploaded file to a local temp path preserving the original name."""
    import uuid
    # Use a project-local temp directory (D: drive) instead of system temp (C: drive)
    # since the C: drive is completely out of space (Errno 28).
    local_tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_uploads")
    os.makedirs(local_tmp_dir, exist_ok=True)
    
    safe_name = f"{uuid.uuid4().hex[:8]}_{file_obj.filename}"
    real_path = os.path.join(local_tmp_dir, safe_name)
    
    file_obj.save(real_path)
    return real_path


# ── Routes ───────────────────────────────────────────────────

@flask_app.route("/")
def index():
    if os.path.exists(os.path.join(FRONTEND_DIST, "index.html")):
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({"error": "React frontend not built yet. Run 'npm run build' in the frontend folder."})

@flask_app.route("/<path:path>")
def serve_static(path):
    # Serve static assets if they exist
    if os.path.exists(os.path.join(FRONTEND_DIST, path)):
        return send_from_directory(FRONTEND_DIST, path)
    # Default to index.html for React routing
    return send_from_directory(FRONTEND_DIST, "index.html")


@flask_app.route("/api/status")
def api_status():
    return jsonify({
        "model_ready":     check_model_ready(),
        "baseline_ready":  check_baseline_ready(),
        "baseline_status": get_baseline_status_text(),
    })


@flask_app.route("/api/diagnose", methods=["POST"])
def api_diagnose():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    tmp_path = save_upload(request.files["audio"])
    try:
        (rf_text, waveform_fig, spectrogram_fig, mel_fig, gauge_fig,
         baseline_text, baseline_gauge, anomaly_fig,
         prob_normal, prob_abnormal, pass_threshold,
         distance, sigma, anomaly_data) = analyze_engine_audio(tmp_path)
        return jsonify({
            "rf_text":        rf_text,
            "baseline_text":  baseline_text,
            "waveform":       fig_to_b64(waveform_fig),
            "spectrogram":    fig_to_b64(spectrogram_fig),
            "mel":            fig_to_b64(mel_fig),
            "gauge":          fig_to_b64(gauge_fig),
            "baseline_gauge": fig_to_b64(baseline_gauge),
            "anomaly":        fig_to_b64(anomaly_fig),
            "prob_normal":    float(prob_normal),
            "prob_abnormal":  float(prob_abnormal),
            "pass_threshold": float(pass_threshold),
            "distance":       float(distance),
            "sigma":          float(sigma),
            "anomaly_data":   anomaly_data,
        })
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@flask_app.route("/api/baseline/status")
def api_baseline_status():
    status, img_path = reload_baseline_status()
    return jsonify({
        "status": status,
        "image":  img_path_to_b64(img_path),
    })


@flask_app.route("/api/baseline/build-folder", methods=["POST"])
def api_baseline_build_folder():
    def generate():
        for log, status, img_path in run_baseline_build():
            data = json.dumps({
                "log":    log,
                "status": status,
                "image":  img_path_to_b64(img_path),
            })
            yield f"data: {data}\n\n"
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@flask_app.route("/api/baseline/build-upload", methods=["POST"])
def api_baseline_build_upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    tmp_paths = [save_upload(f) for f in files]

    def generate():
        try:
            for log, status, img_path in build_baseline_from_uploads(tmp_paths, ""):
                data = json.dumps({
                    "log":    log,
                    "status": status,
                    "image":  img_path_to_b64(img_path),
                })
                yield f"data: {data}\n\n"
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except Exception:
                    pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@flask_app.route("/api/multi/analyse", methods=["POST"])
def api_multi_analyse():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    tmp_paths = [save_upload(f) for f in files]
    try:
        results = analyse_all_files(tmp_paths)
        return _package_multi_results(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {e}"}), 500
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass


@flask_app.route("/api/multi/analyse-folder", methods=["POST"])
def api_multi_analyse_folder():
    body   = request.get_json(silent=True) or {}
    folder = body.get("folder", "").strip()
    if not folder:
        return jsonify({"error": "No folder path provided."}), 400
    try:
        results = load_files_from_folder(folder)
        return _package_multi_results(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {e}"}), 500


def _package_multi_results(results):
    """Serialise the 19-tuple from analyse_all_files / load_files_from_folder."""
    try:
        if len(results) < 19:
            return jsonify({"error": f"Unexpected result length {len(results)}"}), 500

        (status_msg, stats_df, rms_fig, dur_fig, peakfreq_fig, zcr_fig,
         filenames, pie_fig, box_rms, box_pf, box_zcr, box_energy, box_cf,
         box_dur, anomaly_df, scatter_fig, t_test_df, sig_msg,
         metadata_df) = results

        anom_total, anom_count, anom_rate = get_anomaly_metrics(
            pd.DataFrame(df_to_records(anomaly_df)) if anomaly_df is not None else None
        )

        return jsonify({
            "status":          status_msg,
            "stats":           df_to_records(stats_df),
            "filenames":       filenames if isinstance(filenames, list) else [],
            "rms_chart":       fig_to_b64(rms_fig),
            "dur_chart":       fig_to_b64(dur_fig),
            "pf_chart":        fig_to_b64(peakfreq_fig),
            "zcr_chart":       fig_to_b64(zcr_fig),
            "pie_chart":       fig_to_b64(pie_fig),
            "box_rms":         fig_to_b64(box_rms),
            "box_pf":          fig_to_b64(box_pf),
            "box_zcr":         fig_to_b64(box_zcr),
            "box_energy":      fig_to_b64(box_energy),
            "box_cf":          fig_to_b64(box_cf),
            "box_dur":         fig_to_b64(box_dur),
            "anomaly":         df_to_records(anomaly_df),
            "scatter":         fig_to_b64(scatter_fig),
            "ttest":           df_to_records(t_test_df),
            "sig_msg":         sig_msg,
            "metadata":        df_to_records(metadata_df),
            "anomaly_detail":  make_anomaly_detail_text(None),
            "anomaly_metrics": [anom_total, anom_count, anom_rate],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Serialisation failed: {e}"}), 500


@flask_app.route("/api/multi/waveform")
def api_waveform():
    filename = request.args.get("file", "")
    return jsonify({"image": fig_to_b64(view_waveform(filename))})


@flask_app.route("/api/multi/waveform-zoom")
def api_waveform_zoom():
    filename = request.args.get("file", "")
    start    = float(request.args.get("start", 0))
    end      = float(request.args.get("end", 5))
    return jsonify({"image": fig_to_b64(view_waveform_zoomed(filename, start, end))})


@flask_app.route("/api/multi/fft")
def api_fft():
    filename   = request.args.get("file", "")
    freq_limit = int(request.args.get("freq_limit", 5000))
    return jsonify({"image": fig_to_b64(view_fft(filename, freq_limit))})


@flask_app.route("/api/multi/spectrogram")
def api_spectrogram():
    filename = request.args.get("file", "")
    return jsonify({"image": fig_to_b64(view_spectrogram(filename))})


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  ENGINE ACOUSTIC DIAGNOSTIC  —  Starting App")
    print("=" * 55)
    print(f"  RF Model   : {'Ready' if check_model_ready()    else 'Not trained yet'}")
    print(f"  Baseline   : {'Ready' if check_baseline_ready() else 'Not built yet'}")
    print("  Opening app at: http://localhost:5000")
    print("  Serving UI  at: http://localhost:5000/")
    print("=" * 55 + "\n")
    flask_app.run(host="0.0.0.0", port=5000, debug=False)
