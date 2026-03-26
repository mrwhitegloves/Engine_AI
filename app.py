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
import gradio as gr
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from scipy import stats as scipy_stats

from predict import (
    predict_engine_status,
    generate_all_plots,
    format_result_text,
    compute_baseline_distance,
    plot_baseline_comparison,
    plot_worm_graph,
    plot_anomaly_indicators,
    compute_ensemble_status,
    compute_weighted_zscore_vote,
    load_feature_importances,
)
from config import BASELINE_PATH, BASELINE_PLOT, NORMAL_DIR

logo_path = os.path.abspath("ForBlack.png")

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
                None, None, None)

    if not check_model_ready():
        msg = ("RF Model not found!\n\nPlease train first:\n"
               "   python train_model.py\n\nThen restart this app.")
        return msg, None, None, None, None, "RF model not ready.", None, None, None

    try:
        status, confidence, prediction, prob_normal, prob_abnormal, pass_threshold = \
            predict_engine_status(audio_file)
    except Exception as err:
        return (f"Classifier error:\n{err}",
                None, None, None, None, "Classifier failed.", None, None, None)

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

    rf_text = format_result_text(final_status, confidence,
                                  prob_normal, prob_abnormal,
                                  pass_threshold, ensemble_explanation)

    try:
        waveform_fig, spectrogram_fig, mel_fig, gauge_fig = \
            generate_all_plots(audio_file, final_status, prob_normal, prob_abnormal)
    except Exception:
        waveform_fig = spectrogram_fig = mel_fig = gauge_fig = None

    worm_fig = anomaly_fig = None
    if check_baseline_ready():
        try:
            worm_fig = plot_worm_graph(audio_file)
        except Exception:
            worm_fig = None
        try:
            anomaly_fig = plot_anomaly_indicators(audio_file)
        except Exception:
            anomaly_fig = None

    return (rf_text, waveform_fig, spectrogram_fig, mel_fig, gauge_fig,
            baseline_text, baseline_gauge, worm_fig, anomaly_fig)


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
        empty = [None] * 18                # NEW — 18 outputs now (added metadata_table)
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
        gr.Dropdown(choices=filenames, value=filenames[0] if filenames else None),
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
        empty = [None] * 18
        empty[0] = "Enter a folder path and click Load Folder."
        return tuple(empty)

    folder_path = folder_path.strip()
    if not os.path.isdir(folder_path):
        empty = [None] * 18
        empty[0] = f"Folder not found: {folder_path}"
        return tuple(empty)

    wav_files = list(Path(folder_path).glob("*.wav"))
    if not wav_files:
        empty = [None] * 18
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
# BUILD THE GRADIO UI
# ════════════════════════════════════════════════════════════

def build_ui():
    """Construct and return the full Gradio Blocks application."""

    with gr.Blocks(
        title = "Engine Acoustic Diagnostic AI",
        theme = gr.themes.Base(
            primary_hue   = "blue",
            secondary_hue = "slate",
            font          = [gr.themes.GoogleFont("Inter"), "sans-serif"]
        ),
        css = """
            .mono textarea { font-family: monospace !important; font-size: 13px !important; }
            .big-btn { font-size: 16px !important; }
            .stat-table { font-size: 12px !important; }
        """
    ) as app:
        
        with gr.Row():
                gr.Markdown("""
        # Engine Acoustic Intelligence System
        **Three-layer analysis: RF Classifier · Acoustic Baseline · Multi-File Statistics**
        """)

        # ════════════════════════════════════════════════════
        # TAB 1 — ENGINE DIAGNOSIS  (unchanged)
        # ════════════════════════════════════════════════════
        with gr.Tab("Engine Diagnosis"):

            if not check_model_ready():
                gr.Markdown("> RF Model not trained. Run `python train_model.py`, then restart.")
            if not check_baseline_ready():
                gr.Markdown("> No baseline built yet. Use the Baseline Builder tab.")

            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("### Upload Audio")
                    audio_input = gr.Audio(
                        type="filepath", label="Engine Recording (.wav / .mp3)",
                        sources=["upload", "microphone"])
                    analyze_btn = gr.Button("Analyze Engine", variant="primary",
                                            size="lg", elem_classes=["big-btn"])
                    gr.Markdown("**Supported:** WAV, MP3, FLAC, OGG  \n**Ideal:** 10s, 600-750 RPM")

                with gr.Column(scale=2):
                    gr.Markdown("### Classifier Report (Random Forest)")
                    rf_result_output = gr.Textbox(
                        label="Engine Status", lines=12, max_lines=14,
                        elem_classes=["mono"],
                        placeholder="Upload an audio file and click Analyze...")

            gr.Markdown("### Classifier Confidence")
            gauge_output = gr.Plot(label="Normal vs Abnormal Probability")
            gr.Markdown("---")
            gr.Markdown("### Baseline Comparison (Acoustic Fingerprint Distance)")

            with gr.Row():
                with gr.Column(scale=2):
                    baseline_text_output = gr.Textbox(
                        label="Baseline Distance Report", lines=12, max_lines=14,
                        elem_classes=["mono"],
                        placeholder="Baseline comparison will appear here...")
                with gr.Column(scale=3):
                    baseline_gauge_output = gr.Plot(label="Distance from Healthy Baseline")

            gr.Markdown("---")
            gr.Markdown("### Worm Graph — Cumulative Mel-Band Energy Comparison")
            worm_graph_output = gr.Plot(label="Worm Graph (Baseline vs Current Engine)")
            gr.Markdown("---")
            gr.Markdown("### Anomaly Indicators — Per-Feature Deviation")
            anomaly_output = gr.Plot(label="Anomaly Indicators (41 features, z-score coloured)")
            gr.Markdown("---")
            gr.Markdown("### Audio Visual Analysis")
            with gr.Row():
                waveform_output    = gr.Plot(label="Sound Waveform")
                spectrogram_output = gr.Plot(label="Frequency Spectrogram")
            mel_output = gr.Plot(label="Mel Spectrogram")

            analyze_btn.click(
                fn=analyze_engine_audio, inputs=[audio_input],
                outputs=[rf_result_output, waveform_output, spectrogram_output,
                         mel_output, gauge_output, baseline_text_output,
                         baseline_gauge_output, worm_graph_output, anomaly_output])

        # ════════════════════════════════════════════════════
        # TAB 2 — BASELINE BUILDER  (unchanged)
        # ════════════════════════════════════════════════════
        with gr.Tab("Baseline Builder"):

            gr.Markdown("""
            ## Build Acoustic Baseline
            Reads all `.wav` files from `dataset/normal/` and computes the
            **mean acoustic fingerprint** of a healthy engine.
            The result is stored in `models/baseline.pkl` and used automatically in Tab 1.
            """)

            with gr.Row():
                baseline_status_label = gr.Textbox(
                    value=get_baseline_status_text(), label="Current Baseline Status",
                    interactive=False, elem_classes=["mono"])
                refresh_btn = gr.Button("Refresh Status", size="sm")

            # NEW — two ways to build baseline: upload files OR use folder
            gr.Markdown("### Option A — Upload .wav files directly")
            gr.Markdown(
                "*Select multiple healthy engine recordings. "
                "They will be copied to `dataset/normal/` and the baseline will be built automatically.*"
            )
            baseline_upload = gr.File(                     # NEW
                label      = "Upload Healthy Engine .wav Files",
                file_count = "multiple",
                file_types = [".wav"],
            )
            upload_build_btn = gr.Button(                  # NEW
                "Build Baseline from Uploaded Files",
                variant      = "primary",
                size         = "lg",
                elem_classes = ["big-btn"],
            )

            gr.Markdown("### Option B — Use files already in `dataset/normal/`")
            with gr.Row():
                with gr.Column(scale=1):
                    build_btn = gr.Button("Build Baseline from dataset/normal/",
                                          variant="secondary", size="lg",
                                          elem_classes=["big-btn"])
                    gr.Markdown("""
                    **Before clicking Option B:**
                    - Add `.wav` files to `dataset/normal/`
                    - Aim for 10+ healthy recordings

                    **Files saved either way:**
                    - `models/baseline.pkl`
                    - `models/baseline_features.csv`
                    - `models/avg_spectrogram.png`
                    """)
                with gr.Column(scale=2):
                    build_log_output = gr.Textbox(
                        label="Build Log", lines=15, max_lines=20,
                        elem_classes=["mono"],
                        placeholder="Click either Build button to start...")

            gr.Markdown("### Average Spectrogram of Healthy Engine Baseline")
            spectrogram_image = gr.Image(
                value=BASELINE_PLOT if os.path.exists(BASELINE_PLOT) else None,
                label="Average Mel-Spectrogram", type="filepath", height=380)

            # Wire Option A (upload files → build)                # NEW
            upload_build_btn.click(                               # NEW
                fn      = build_baseline_from_uploads,           # NEW
                inputs  = [baseline_upload, build_log_output],   # NEW
                outputs = [build_log_output, baseline_status_label, spectrogram_image],  # NEW
            )                                                     # NEW
            # Wire Option B (folder → build)
            build_btn.click(fn=run_baseline_build, inputs=[],
                            outputs=[build_log_output, baseline_status_label, spectrogram_image])
            refresh_btn.click(fn=reload_baseline_status, inputs=[],
                              outputs=[baseline_status_label, spectrogram_image])

        # ════════════════════════════════════════════════════
        # TAB 3 — MULTI-FILE AUDIO ANALYSIS  (ALL NEW)
        # ════════════════════════════════════════════════════
        with gr.Tab("Multi-File Analysis"):

            gr.Markdown("""
            ## WAV Audio Analysis Dashboard
            Upload multiple engine WAV files to get statistical comparisons, frequency
            analysis, spectrograms, group comparison, and anomaly detection — just like
            the standalone Streamlit dashboard, but integrated directly into this app.

            **How files are classified:**
            Files whose names contain `abnormal / damper / defective / cam / valvelash /
            bad / fault / fail / noise / knock` are labelled **Bad Signals**.
            All others are labelled **Good Signals**.
            """)

            # ── Input Method: Upload OR Folder Path ──────────
            gr.Markdown("### Input Method")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Option A — Upload files directly**")
                    multi_upload = gr.File(
                        label       = "Upload WAV Files (select multiple)",
                        file_count  = "multiple",
                        file_types  = [".wav"],
                    )
                    analyse_all_btn = gr.Button(
                        "Analyse Uploaded Files",
                        variant      = "primary",
                        size         = "lg",
                        elem_classes = ["big-btn"],
                    )

                with gr.Column(scale=1):
                    gr.Markdown("**Option B — Load from folder path**")        # NEW
                    folder_path_input = gr.Textbox(                             # NEW
                        label       = "Folder path containing .wav files",
                        placeholder = r"e.g.  C:\Users\you\engine_sounds",  # NEW
                        lines       = 1,
                    )
                    load_folder_btn = gr.Button(                                # NEW
                        "Load Folder & Analyse",
                        variant      = "secondary",
                        size         = "lg",
                        elem_classes = ["big-btn"],
                    )
                    gr.Markdown("""
                    **Tips:**
                    - Select all files at once (Ctrl+click or Shift+click)
                    - Only `.wav` files are supported
                    - Files are auto-classified by filename keywords
                    """)

            with gr.Row():
                with gr.Column(scale=2):
                    multi_status = gr.Textbox(
                        label       = "Status",
                        lines       = 3,
                        interactive = False,
                        placeholder = "Upload files or enter folder path and click Analyse...",
                    )
                    # Stats table shown immediately after analysis
                    multi_stats_df = gr.DataFrame(
                        label             = "Statistical Summary (all files)",
                        elem_classes      = ["stat-table"],
                        wrap              = True,
                        max_height        = 300,
                    )
                    stats_download_btn = gr.Button("Download Statistics CSV", size="sm")

            # ── SECTION 1: Overview Bar Charts ────────────────
            gr.Markdown("---")
            gr.Markdown("### Overview Statistics")

            with gr.Row():
                rms_chart      = gr.Plot(label="RMS Amplitude by File")
                duration_chart = gr.Plot(label="Duration (s) by File")
            with gr.Row():
                peakfreq_chart = gr.Plot(label="Peak Frequency (Hz) by File")
                zcr_chart      = gr.Plot(label="Zero Crossings by File")

            # ── SECTION 2: Waveform Viewer ────────────────────
            gr.Markdown("---")
            gr.Markdown("### Waveform Viewer")

            waveform_file_dropdown = gr.Dropdown(
                label   = "Select file to view waveform",
                choices = [],
            )
            waveform_plot = gr.Plot(label="Waveform (full)")

            # NEW — zoom controls (mirrors Streamlit time-range sliders)
            gr.Markdown("**Zoom to time range:**")             # NEW
            with gr.Row():                                      # NEW
                zoom_start = gr.Slider(                         # NEW
                    label="Start time (s)", minimum=0, maximum=30,
                    value=0, step=0.1)
                zoom_end   = gr.Slider(                         # NEW
                    label="End time (s)",   minimum=0, maximum=30,
                    value=5, step=0.1)
            zoomed_waveform_plot = gr.Plot(label="Zoomed Waveform")  # NEW

            # # ── SECTION 3: Frequency Analysis (FFT) ──────────
            # gr.Markdown("---")
            # gr.Markdown("### Frequency Analysis (FFT)")

            # fft_file_dropdown = gr.Dropdown(
            #     label   = "Select file for FFT",
            #     choices = [],
            # )
            # freq_limit_slider = gr.Slider(
            #     label   = "Frequency limit (Hz)",
            #     minimum = 100,
            #     maximum = 20000,
            #     value   = 5000,
            #     step    = 100,
            # )
            # fft_plot = gr.Plot(label="FFT Spectrum")

            # # ── SECTION 4: Spectrogram ────────────────────────
            # gr.Markdown("---")
            # gr.Markdown("### Spectrogram Analysis")

            # spec_file_dropdown = gr.Dropdown(
            #     label   = "Select file for spectrogram",
            #     choices = [],
            # )
            # spectrogram_plot = gr.Plot(label="Spectrogram (dB)")

            # ── SECTION 5: Group Comparison ───────────────────
            gr.Markdown("---")
            gr.Markdown("### Group Comparison (Good Signals vs Bad Signals)")

            with gr.Row():
                pie_chart    = gr.Plot(label="Group Distribution")
                box_rms      = gr.Plot(label="RMS Distribution")
            with gr.Row():
                box_pf       = gr.Plot(label="Peak Frequency Distribution")
                box_zcr      = gr.Plot(label="Zero Crossings Distribution")
            with gr.Row():
                box_energy   = gr.Plot(label="Energy Distribution")
                box_cf       = gr.Plot(label="Crest Factor Distribution")
            box_dur          = gr.Plot(label="Duration Distribution")

            # ── SECTION 6: Anomaly Detection ──────────────────
            gr.Markdown("---")
            gr.Markdown("""
            ### Anomaly Detection
            *Thresholds are computed from Good Signals (mean ± 2 std).
            Any file outside this range for a given metric is flagged as anomalous.*
            """)

            # NEW — anomaly count metrics (mirrors Streamlit metric boxes)
            gr.Markdown("**Anomaly Summary:**")                # NEW
            with gr.Row():                                      # NEW
                metric_total     = gr.Textbox(label="Total Files",      interactive=False, lines=1)  # NEW
                metric_anomalous = gr.Textbox(label="Anomalous Files",  interactive=False, lines=1)  # NEW
                metric_rate      = gr.Textbox(label="Anomaly Rate (%)", interactive=False, lines=1)  # NEW

            anomaly_summary_df = gr.DataFrame(
                label        = "Anomaly Detection Results (per file)",
                max_height   = 300,
            )
            anomaly_scatter    = gr.Plot(label="Anomaly Scatter: RMS vs Peak Frequency")

            # NEW — per-file detailed anomaly breakdown (mirrors Streamlit expanders)
            gr.Markdown("**Detailed Anomaly Breakdown** *(files outside mean ± 2std for any metric):*")  # NEW
            anomaly_detail_text = gr.Textbox(                  # NEW
                label       = "Per-File Anomaly Details",
                lines       = 12,
                max_lines   = 20,
                interactive = False,
                elem_classes= ["mono"],
                placeholder = "Run analysis to see per-file anomaly detail...",
            )

            ttest_df_output    = gr.DataFrame(
                label        = "T-Test Significance Table (Good vs Bad Signals)",
                max_height   = 260,
            )
            significance_msg   = gr.Textbox(
                label       = "Significance Summary",
                lines       = 2,
                interactive = False,
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Bad Signals (Anomalies)**")
                    bad_files_list = gr.Textbox(
                        label="Files classified as Bad Signals",
                        lines=8, interactive=False)
                with gr.Column():
                    gr.Markdown("**Good Signals (Normal)**")
                    good_files_list = gr.Textbox(
                        label="Files classified as Good Signals",
                        lines=8, interactive=False)

            # NEW — Metadata Analysis section (mirrors Streamlit Tab 5)
            gr.Markdown("---")                                 # NEW
            gr.Markdown("### Metadata Analysis (Filename Parts)")  # NEW
            gr.Markdown(                                        # NEW
                "*Each filename is split on underscores. "
                "Part_1, Part_2, ... show the individual tokens "
                "so you can identify engine type, serial number, date, etc.*"
            )
            metadata_table_output = gr.DataFrame(              # NEW
                label      = "Filename Metadata (underscore-split)",
                max_height = 350,
            )

            # ── Wire: Analyse All Files button ────────────────
            analyse_outputs = [
                multi_status,          # 1
                multi_stats_df,        # 2
                rms_chart,             # 3
                duration_chart,        # 4
                peakfreq_chart,        # 5
                zcr_chart,             # 6
                waveform_file_dropdown,# 7  (gr.Dropdown update)
                pie_chart,             # 8
                box_rms,               # 9
                box_pf,                # 10
                box_zcr,               # 11
                box_energy,            # 12
                box_cf,                # 13
                box_dur,               # 14
                anomaly_summary_df,    # 15
                anomaly_scatter,       # 16
                ttest_df_output,       # 17
                significance_msg,      # 18
                metadata_table_output, # 19 NEW
            ]

            def analyse_and_update_dropdowns(files):
                """
                Wrapper that calls analyse_all_files and also populates
                the FFT and Spectrogram dropdowns (which aren't in the
                main return list to keep things clean).
                """
                results = analyse_all_files(files)
                # results[6] is the gr.Dropdown update for waveform_file_dropdown
                return results

            analyse_all_btn.click(
                fn      = analyse_and_update_dropdowns,
                inputs  = [multi_upload],
                outputs = analyse_outputs,
            )

            # ── After analysis, sync the FFT and spectrogram dropdowns ──
            # We do this by updating them whenever waveform_file_dropdown changes
            def sync_all_dropdowns(dropdown_update):
                """When the waveform dropdown is refreshed, sync FFT + spectrogram."""
                choices = dropdown_update.get("choices", []) if isinstance(dropdown_update, dict) else []
                value   = choices[0] if choices else None
                return (gr.Dropdown(choices=choices, value=value),
                        gr.Dropdown(choices=choices, value=value))

            waveform_file_dropdown.change(
                fn      = sync_all_dropdowns,
                inputs  = [waveform_file_dropdown],
                # outputs = [fft_file_dropdown, spec_file_dropdown],
            )

            # ── Wire: per-file view callbacks ─────────────────
            waveform_file_dropdown.change(
                fn=view_waveform, inputs=[waveform_file_dropdown],
                outputs=[waveform_plot])

            # fft_file_dropdown.change(
            #     fn=view_fft,
            #     inputs=[fft_file_dropdown, freq_limit_slider],
            #     outputs=[fft_plot])

            # freq_limit_slider.change(
            #     fn=view_fft,
            #     inputs=[fft_file_dropdown, freq_limit_slider],
            #     outputs=[fft_plot])

            # spec_file_dropdown.change(
            #     fn=view_spectrogram, inputs=[spec_file_dropdown],
            #     outputs=[spectrogram_plot])

            # ── Wire: bad/good file lists ─────────────────────
            def update_file_lists(anomaly_df):
                if anomaly_df is None or len(anomaly_df) == 0:
                    return "", ""
                df = _DF_CACHE.get("df")
                if df is None:
                    return "", ""
                bad_files  = df[df["Group"] == "Bad Signals"]["Filename"].tolist()
                good_files = df[df["Group"] == "Good Signals"]["Filename"].tolist()
                return "\n".join(bad_files) or "None", "\n".join(good_files) or "None"

            anomaly_summary_df.change(
                fn=update_file_lists, inputs=[anomaly_summary_df],
                outputs=[bad_files_list, good_files_list])

            # NEW — Wire: anomaly metrics from anomaly_df
            def update_anomaly_metrics(anomaly_df):             # NEW
                return get_anomaly_metrics(anomaly_df)          # NEW

            anomaly_summary_df.change(                          # NEW
                fn      = update_anomaly_metrics,               # NEW
                inputs  = [anomaly_summary_df],                 # NEW
                outputs = [metric_total, metric_anomalous, metric_rate],  # NEW
            )

            # NEW — Wire: anomaly detail text (triggered by anomaly_df change)
            anomaly_summary_df.change(                          # NEW
                fn      = make_anomaly_detail_text,             # NEW
                inputs  = [anomaly_summary_df],                 # NEW
                outputs = [anomaly_detail_text],                # NEW
            )

            # NEW — Wire: zoom sliders → zoomed waveform
            zoom_start.change(                                  # NEW
                fn=view_waveform_zoomed,
                inputs=[waveform_file_dropdown, zoom_start, zoom_end],
                outputs=[zoomed_waveform_plot])
            zoom_end.change(                                    # NEW
                fn=view_waveform_zoomed,
                inputs=[waveform_file_dropdown, zoom_start, zoom_end],
                outputs=[zoomed_waveform_plot])
            waveform_file_dropdown.change(                      # NEW (also update zoom plot)
                fn=view_waveform_zoomed,
                inputs=[waveform_file_dropdown, zoom_start, zoom_end],
                outputs=[zoomed_waveform_plot])

            # NEW — Wire: folder path load button
            load_folder_btn.click(                              # NEW
                fn      = load_files_from_folder,              # NEW
                inputs  = [folder_path_input],                 # NEW
                outputs = analyse_outputs,                     # NEW (same outputs as upload)
            )

            # # ── Wire: CSV download ────────────────────────────
            # def download_csv():
            #     df = _DF_CACHE.get("df")
            #     if df is None:
            #         return None
            #     path = "models/audio_statistics.csv"
            #     df.to_csv(path, index=False)
            #     return path

            # stats_download_btn.click(
            #     fn=download_csv, inputs=[], outputs=[gr.File(label="Download CSV")])

        gr.Markdown("---\n*Engine Acoustic Diagnostic AI — EngenX*")

    return app


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  ENGINE ACOUSTIC DIAGNOSTIC  —  Starting App")
    print("=" * 55)
    print(f"  RF Model   : {'Ready' if check_model_ready()    else 'Not trained yet'}")
    print(f"  Baseline   : {'Ready' if check_baseline_ready() else 'Not built yet'}")
    print("\n  Opening app at: http://localhost:7860")
    print("=" * 55 + "\n")
    app = build_ui()
    app.launch(
        server_name="0.0.0.0", 
        server_port = int(os.environ.get("PORT", 7860)), 
        share=False, 
        show_error=True,
    )