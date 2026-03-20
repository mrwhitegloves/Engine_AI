# ============================================================
# app.py  —  Gradio web interface for the Engine Diagnostic AI
#
# Now has TWO TABS:
#   Tab 1 — 🔍 Engine Diagnosis    (RF classifier + baseline comparison)
#   Tab 2 — 📐 Baseline Builder    (build / rebuild baseline from audio)
#
# Run:   python app.py
# Open:  http://localhost:7860
# ============================================================

import os
import gradio as gr

from predict import (
    predict_engine_status,
    generate_all_plots,
    format_result_text,
    compute_baseline_distance,
    plot_baseline_comparison,
    plot_worm_graph,
    plot_anomaly_indicators,
    compute_ensemble_status,       # CHANGED — 3-detector voting
    compute_weighted_zscore_vote,  # CHANGED — z-score detector
    load_feature_importances,      # CHANGED — feature weights
)
from config import BASELINE_PATH, BASELINE_PLOT, NORMAL_DIR


# ────────────────────────────────────────────────────────────
# STATUS CHECKERS
# ────────────────────────────────────────────────────────────
def check_model_ready():
    """Return True only if the RF model files exist."""
    return (
        os.path.exists("models/engine_model.pkl") and
        os.path.exists("models/scaler.pkl")
    )


def check_baseline_ready():
    """Return True only if baseline.pkl exists."""
    return os.path.exists(BASELINE_PATH)


def get_baseline_status_text():
    """Return a short status string shown in the UI."""
    if check_baseline_ready():
        import joblib
        try:
            data = joblib.load(BASELINE_PATH)
            n    = data.get("n_samples", "?")
            return f"✅  Baseline ready — built from {n} healthy sample(s)"
        except Exception:
            return "✅  Baseline file exists (could not read details)"
    return "⚠️  No baseline built yet — use the Baseline Builder tab"


# ════════════════════════════════════════════════════════════
# TAB 1 FUNCTIONS  —  Engine Diagnosis
# ════════════════════════════════════════════════════════════

def analyze_engine_audio(audio_file):
    """
    Called when the user clicks 'Analyze Engine' in Tab 1.

    Runs BOTH systems:
      1. Random Forest classifier  →  PASS / FAIL + confidence
      2. Baseline comparison       →  distance from healthy fingerprint

    Returns (must match outputs list in build_ui — 9 values):
      rf_report_text        str
      waveform_fig          Figure
      spectrogram_fig       Figure
      mel_fig               Figure
      gauge_fig             Figure
      baseline_report_text  str
      baseline_gauge_fig    Figure
      worm_graph_fig        Figure   ← NEW
      anomaly_fig           Figure   ← NEW
    """
    # ── Guard: no file ───────────────────────────────────────
    if audio_file is None:
        return (
            "⚠️  Please upload an engine audio file first.",
            None, None, None, None,
            "⚠️  Upload a file to see baseline comparison.",
            None, None, None
        )

    # ── Guard: RF model missing ──────────────────────────────
    if not check_model_ready():
        msg = (
            "❌  RF Model not found!\n\n"
            "Please train first:\n"
            "   python train_model.py\n\n"
            "Then restart this app."
        )
        return msg, None, None, None, None, "RF model not ready.", None, None, None

    # ── Step 1: RF classifier (now returns 6 values) ────────
    try:
        # CHANGED — predict_engine_status now also returns pass_threshold
        status, confidence, prediction, prob_normal, prob_abnormal, pass_threshold = \
            predict_engine_status(audio_file)
    except Exception as err:
        return (
            f"Classifier error:\n{err}",
            None, None, None, None,
            "Classifier failed.", None, None, None
        )

    # ── Step 2: Baseline comparison ──────────────────────────
    risk_level     = "N/A"
    baseline_text  = ""
    baseline_gauge = None
    sigma          = 0.0

    if check_baseline_ready():
        try:
            distance, sigma, risk_level, baseline_text = \
                compute_baseline_distance(audio_file)
            baseline_gauge = plot_baseline_comparison(distance, sigma, risk_level)
        except Exception as err:
            baseline_text  = f"Baseline comparison error:\n{err}"
            baseline_gauge = None
            risk_level     = "N/A"
    else:
        baseline_text = (
            "No baseline built yet.\n\n"
            "Go to the Baseline Builder tab and click\n"
            "Build Baseline to enable this analysis."
        )

    # ── Step 3: Z-score vote (Task 2) ────────────────────────
    # CHANGED — compute independent z-score detector
    try:
        importances  = load_feature_importances()
        zscore_vote, zscore_score = compute_weighted_zscore_vote(audio_file, importances)
    except Exception:
        zscore_vote, zscore_score = "PASS", 0.0       # CHANGED — safe default

    # ── Step 4: Ensemble voting (Task 6 & 10) ────────────────
    # CHANGED — combine RF + baseline + z-score for final status
    try:
        final_status, ensemble_explanation = compute_ensemble_status(
            prob_normal, risk_level, zscore_vote
        )
    except Exception:
        final_status          = status                 # CHANGED — fallback to RF
        ensemble_explanation  = f"(ensemble error, using RF only)"

    # ── Step 5: Build text report with ensemble result ────────
    # CHANGED — format_result_text now accepts pass_threshold + explanation
    rf_text = format_result_text(
        final_status, confidence,                      # CHANGED — use ensemble status
        prob_normal, prob_abnormal,
        pass_threshold,
        ensemble_explanation
    )

    # ── Step 6: Generate visual plots ────────────────────────
    try:
        waveform_fig, spectrogram_fig, mel_fig, gauge_fig = \
            generate_all_plots(audio_file, final_status,  # CHANGED — final_status colours
                               prob_normal, prob_abnormal)
    except Exception:
        waveform_fig = spectrogram_fig = mel_fig = gauge_fig = None

    # ── Step 7: Worm graph & anomaly indicators ───────────────
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

    return (
        rf_text,
        waveform_fig,
        spectrogram_fig,
        mel_fig,
        gauge_fig,
        baseline_text,
        baseline_gauge,
        worm_fig,
        anomaly_fig,
    )


# ════════════════════════════════════════════════════════════
# TAB 2 FUNCTIONS  —  Baseline Builder
# ════════════════════════════════════════════════════════════

def run_baseline_build():
    """
    Called when the user clicks 'Build Baseline' in Tab 2.
    Yields incremental log updates back to the UI.
    """
    logs = []

    def log(msg):
        logs.append(str(msg))

    yield "\n".join(logs), "⏳ Building baseline ...", None

    try:
        from acoustic_baseline import build_acoustic_baseline
        build_acoustic_baseline(audio_folder=NORMAL_DIR, log_fn=log)
        yield "\n".join(logs), get_baseline_status_text(), \
              BASELINE_PLOT if os.path.exists(BASELINE_PLOT) else None

    except FileNotFoundError as err:
        logs.append(f"\n❌  {err}")
        yield "\n".join(logs), "❌  Build failed — see log", None

    except Exception as err:
        logs.append(f"\n❌  Unexpected error: {err}")
        yield "\n".join(logs), "❌  Build failed — see log", None


def reload_baseline_status():
    """Called when user clicks Refresh Status in Tab 2."""
    img = BASELINE_PLOT if os.path.exists(BASELINE_PLOT) else None
    return get_baseline_status_text(), img


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
        """
    ) as app:

        gr.Markdown("""
        # 🔊 AI Engine Acoustic Diagnostic System
        **Two-layer analysis: Random Forest Classifier + Acoustic Baseline Comparison**
        """)

        # ════════════════════════════════════════════════════
        # TAB 1 — ENGINE DIAGNOSIS
        # ════════════════════════════════════════════════════
        with gr.Tab("🔍 Engine Diagnosis"):

            if not check_model_ready():
                gr.Markdown("> ⚠️ **RF Model not trained.** Run `python train_model.py`, then restart.")
            if not check_baseline_ready():
                gr.Markdown("> ℹ️ **No baseline built yet.** Use the 📐 Baseline Builder tab to enable baseline comparison.")

            # Upload + RF result
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("### 📁 Upload Audio")
                    audio_input = gr.Audio(
                        type    = "filepath",
                        label   = "Engine Recording (.wav / .mp3)",
                        sources = ["upload", "microphone"]
                    )
                    analyze_btn = gr.Button(
                        "🔍  Analyze Engine",
                        variant      = "primary",
                        size         = "lg",
                        elem_classes = ["big-btn"]
                    )
                    gr.Markdown("**Supported:** WAV, MP3, FLAC, OGG  \n**Ideal:** 10s, 600–750 RPM, no load")

                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Classifier Report  *(Random Forest)*")
                    rf_result_output = gr.Textbox(
                        label        = "Engine Status",
                        lines        = 12,
                        max_lines    = 14,
                        elem_classes = ["mono"],
                        placeholder  = "Upload an audio file and click Analyze..."
                    )

            # RF confidence gauge
            gr.Markdown("### 📊 Classifier Confidence")
            gauge_output = gr.Plot(label="Normal vs Abnormal Probability")

            # ── NEW: Baseline comparison section ─────────────
            gr.Markdown("---")
            gr.Markdown("### 📐 Baseline Comparison  *(Acoustic Fingerprint Distance)*")

            with gr.Row():
                with gr.Column(scale=2):
                    baseline_text_output = gr.Textbox(
                        label        = "Baseline Distance Report",
                        lines        = 12,
                        max_lines    = 14,
                        elem_classes = ["mono"],
                        placeholder  = "Baseline comparison will appear here after analysis..."
                    )
                with gr.Column(scale=3):
                    baseline_gauge_output = gr.Plot(label="Distance from Healthy Baseline (σ gauge)")

            # ── NEW: Worm Graph ───────────────────────────────
            gr.Markdown("---")
            gr.Markdown(
                "### 🐛 Worm Graph — Cumulative Mel-Band Energy Comparison\n"
                "*Like the cricket worm chart: X = frequency bands (low→high), "
                "Y = cumulative energy. "
                "Blue = Healthy Baseline · Grey = Current Engine · "
                "🔴 Red dots = anomaly bands.*"
            )
            worm_graph_output = gr.Plot(label="Worm Graph (Baseline vs Current Engine)")

            # ── NEW: Anomaly Indicators ───────────────────────
            gr.Markdown("---")
            gr.Markdown(
                "### 🔴 Anomaly Indicators — Per-Feature Deviation\n"
                "*Each bar = one acoustic feature. "
                "🟢 Green = normal · 🟡 Orange = warning · 🔴 Red = alert. "
                "Dashed lines mark the ±2σ and ±3.5σ thresholds.*"
            )
            anomaly_output = gr.Plot(label="Anomaly Indicators (41 features, z-score coloured)")

            # Visual plots
            gr.Markdown("---")
            gr.Markdown("### 🎨 Audio Visual Analysis")
            with gr.Row():
                waveform_output    = gr.Plot(label="Sound Waveform")
                spectrogram_output = gr.Plot(label="Frequency Spectrogram")
            mel_output = gr.Plot(label="Mel Spectrogram")

            # Wire button — 9 outputs total
            analyze_btn.click(
                fn      = analyze_engine_audio,
                inputs  = [audio_input],
                outputs = [
                    rf_result_output,
                    waveform_output,
                    spectrogram_output,
                    mel_output,
                    gauge_output,
                    baseline_text_output,
                    baseline_gauge_output,
                    worm_graph_output,      # NEW
                    anomaly_output,         # NEW
                ]
            )

            with gr.Accordion("ℹ️  How This System Works", open=False):
                gr.Markdown("""
                ### Two Analysis Layers

                | Layer | Method | Features | Needs |
                |-------|--------|----------|-------|
                | **Layer 1** | Random Forest | 226 features | Labelled PASS/FAIL data |
                | **Layer 2** | Baseline Distance | 41 features | Healthy recordings only |

                ### 🐛 Worm Graph Explained
                Inspired by the cricket worm chart. Each engine recording is converted to
                128 mel-frequency bands.  Cumulative energy is plotted left→right (low→high frequency).
                - **Blue line** = Healthy baseline (how a good engine should look)
                - **Grey line** = Current engine being tested
                - **🔴 Red dots** = Frequency bands where current engine deviates significantly
                - **Red shading** = Anomaly regions

                ### 🔴 Anomaly Indicators Explained
                Each of the 41 acoustic features gets a z-score: how many standard deviations
                it is from the healthy mean.
                - 🟢 Green bars = within normal variation (|z| < 2σ)
                - 🟡 Orange bars = warning zone (2σ ≤ |z| < 3.5σ)
                - 🔴 Red bars + 🔴 emoji = alert zone (|z| ≥ 3.5σ)

                ### Sigma (σ) Thresholds
                | Sigma | Meaning |
                |-------|---------|
                | σ < 2.0 | 🟢 OK — within healthy variation |
                | 2.0 ≤ σ < 3.5 | 🟡 WARNING — noticeably different |
                | σ ≥ 3.5 | 🔴 ALERT — strongly deviates |

                Tune thresholds in `config.py`: `BASELINE_WARN_SIGMA` and `BASELINE_FAIL_SIGMA`
                """)

        # ════════════════════════════════════════════════════
        # TAB 2 — BASELINE BUILDER
        # ════════════════════════════════════════════════════
        with gr.Tab("📐 Baseline Builder"):

            gr.Markdown("""
            ## Build Acoustic Baseline
            Reads all `.wav` files from `dataset/normal/` and computes the
            **mean acoustic fingerprint** of a healthy engine.  
            The result is stored in `models/baseline.pkl` and used automatically in Tab 1.
            """)

            with gr.Row():
                baseline_status_label = gr.Textbox(
                    value        = get_baseline_status_text(),
                    label        = "Current Baseline Status",
                    interactive  = False,
                    elem_classes = ["mono"]
                )
                refresh_btn = gr.Button("🔄  Refresh Status", size="sm")

            with gr.Row():
                with gr.Column(scale=1):
                    build_btn = gr.Button(
                        "🏗️  Build Baseline from dataset/normal/",
                        variant      = "primary",
                        size         = "lg",
                        elem_classes = ["big-btn"]
                    )
                    gr.Markdown("""
                    **Before clicking:**
                    - Add `.wav` files to `dataset/normal/`
                    - Aim for 10+ healthy recordings

                    **Files saved:**
                    - `models/baseline.pkl`
                    - `models/baseline_features.csv`
                    - `models/avg_spectrogram.png`
                    """)

                with gr.Column(scale=2):
                    build_log_output = gr.Textbox(
                        label        = "Build Log",
                        lines        = 15,
                        max_lines    = 20,
                        elem_classes = ["mono"],
                        placeholder  = "Click 'Build Baseline' to start..."
                    )

            gr.Markdown("### 📊 Average Spectrogram of Healthy Engine Baseline")
            spectrogram_image = gr.Image(
                value   = BASELINE_PLOT if os.path.exists(BASELINE_PLOT) else None,
                label   = "Average Mel-Spectrogram (models/avg_spectrogram.png)",
                type    = "filepath",
                height  = 380
            )

            build_btn.click(
                fn      = run_baseline_build,
                inputs  = [],
                outputs = [build_log_output, baseline_status_label, spectrogram_image]
            )
            refresh_btn.click(
                fn      = reload_baseline_status,
                inputs  = [],
                outputs = [baseline_status_label, spectrogram_image]
            )

            with gr.Accordion("ℹ️  What is an Acoustic Baseline?", open=False):
                gr.Markdown("""
                A baseline is the **average acoustic fingerprint** of a healthy engine.

                **How comparison works:**
                ```
                z_score  = (test_features - baseline_mean) / baseline_std
                distance = L2-norm(z_score)
                sigma    = (distance - healthy_mean) / healthy_std
                ```
                Sigma tells you: *"How many standard deviations is this engine  
                away from the typical healthy engine?"*

                Tune thresholds in `config.py`:
                ```python
                BASELINE_WARN_SIGMA = 2.0   # yellow warning
                BASELINE_FAIL_SIGMA = 3.5   # red alert
                ```
                """)

        gr.Markdown("---\n*Engine Acoustic Diagnostic AI — Librosa · Scikit-learn · Gradio*")

    return app


# ────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  ENGINE ACOUSTIC DIAGNOSTIC  —  Starting App")
    print("=" * 55)
    print(f"  RF Model   : {'✅ Ready' if check_model_ready()    else '⚠️  Not trained yet'}")
    print(f"  Baseline   : {'✅ Ready' if check_baseline_ready() else '⚠️  Not built yet'}")
    print("\n  Opening app at: http://localhost:7860")
    print("=" * 55 + "\n")

    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)