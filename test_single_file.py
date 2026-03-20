# ============================================================
# test_single_file.py  —  Quickly test one audio file from
#                          the command line (no UI needed)
#
# Usage:
#   python test_single_file.py path/to/engine_recording.wav
# ============================================================

import sys
import os

def test_audio_file(file_path):
    """
    Run a prediction on a single audio file and print the result.

    Parameters
    ----------
    file_path : str   Path to the .wav or .mp3 file to test
    """
    # Validate file exists
    if not os.path.exists(file_path):
        print(f"❌  File not found: {file_path}")
        return

    print(f"\n🔍  Analyzing: {os.path.basename(file_path)}")
    print("-" * 50)

    # Import here so errors are clearer
    from predict import predict_engine_status, format_result_text

    try:
        status, confidence, prediction, prob_normal, prob_abnormal = \
            predict_engine_status(file_path)

        result = format_result_text(status, confidence, prob_normal, prob_abnormal)
        print(result)

    except FileNotFoundError as e:
        print(f"\n❌  {e}")
        print("Run 'python train_model.py' first to create the model.")
    except Exception as e:
        print(f"\n❌  Error: {e}")


def run_batch_test(folder_path, expected_label=None):
    """
    Test all audio files in a folder.

    Parameters
    ----------
    folder_path    : str   Folder containing .wav / .mp3 files
    expected_label : str   "normal" or "abnormal" (optional, for accuracy check)
    """
    import glob
    from predict import predict_engine_status

    files = []
    for ext in ["*.wav", "*.mp3", "*.flac"]:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not files:
        print(f"No audio files found in: {folder_path}")
        return

    print(f"\n🔬  Batch Testing: {len(files)} files in '{folder_path}'")
    print("="*60)

    correct = 0
    for file_path in files:
        try:
            status, confidence, prediction, _, _ = predict_engine_status(file_path)
            icon = "✅" if status == "PASS" else "❌"
            print(f"  {icon} {status:4s}  ({confidence:5.1f}%)  {os.path.basename(file_path)}")

            if expected_label:
                if (expected_label == "normal" and status == "PASS") or \
                   (expected_label == "abnormal" and status == "FAIL"):
                    correct += 1
        except Exception as e:
            print(f"  ⚠️  Error — {os.path.basename(file_path)}: {e}")

    if expected_label:
        accuracy = correct / len(files) * 100
        print(f"\n  Accuracy on '{expected_label}' folder: {accuracy:.1f}%  ({correct}/{len(files)})")
    print("="*60)


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage examples:")
        print("  Test a single file:")
        print("    python test_single_file.py path/to/recording.wav")
        print("\n  Batch test a folder:")
        print("    python test_single_file.py path/to/folder/ normal")
        print("    python test_single_file.py path/to/folder/ abnormal")
        sys.exit(0)

    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) >= 3 else None

    if os.path.isdir(path):
        run_batch_test(path, expected_label=label)
    else:
        test_audio_file(path)
