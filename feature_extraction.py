# ============================================================
# feature_extraction.py  —  Convert audio -> numbers the AI
#                            model can understand
#
# CHANGES vs previous version (all marked # CHANGED):
#   Task 9  — normalize_audio()        : peak + RMS normalization
#   Task 4  — apply_bandpass_filter()  : focus on 500-2000 Hz
#   Task 5  — smooth_features()        : moving-average spike removal
#   load_audio()                       : now calls normalize_audio
#   extract_mel_spectrogram_features() : now applies smoothing
#   extract_features()                 : now applies bandpass filter
# ============================================================

import librosa
import numpy as np
from scipy.signal import butter, sosfilt             # CHANGED — new import for bandpass
from config import (
    SAMPLE_RATE, DURATION, N_MFCC, N_MELS,
    HOP_LENGTH, N_FFT,
    BANDPASS_LOW, BANDPASS_HIGH, SMOOTHING_WIN        # CHANGED — new config imports
)


# ────────────────────────────────────────────────────────────
# 0.  AUDIO PRE-PROCESSING HELPERS  (all NEW — Task 4, 5, 9)
# ────────────────────────────────────────────────────────────

def normalize_audio(audio):
    """
    CHANGED (Task 9) — Combined peak + RMS normalization.

    Step 1: Peak-normalize to [-1, +1]  (removes recording level differences)
    Step 2: RMS-normalize to target RMS (removes loudness differences between
            recordings made at different microphone distances)

    Why this matters:
        A recording made 10 cm from the engine vs 50 cm will have very
        different amplitudes.  Without normalization, the RMS energy feature
        alone can flip the classifier from PASS to FAIL.
    """
    # Step 1 — peak normalize
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        return audio                                  # silent file — return as-is
    audio = audio / peak

    # Step 2 — RMS normalize to target RMS = 0.1
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    target_rms = 0.1
    audio = audio * (target_rms / rms)                # CHANGED — new RMS step

    # Clip to [-1, +1] after RMS scaling to prevent overflow
    audio = np.clip(audio, -1.0, 1.0)                 # CHANGED
    return audio.astype(np.float32)


def apply_bandpass_filter(audio, sr, low_hz=BANDPASS_LOW, high_hz=BANDPASS_HIGH):
    """
    CHANGED (Task 4) — Butterworth bandpass filter focused on engine frequencies.

    Engine knock, misfire, and rattle signatures are strongest in the
    500–2000 Hz band.  Frequencies outside this range are mostly:
      < 500 Hz : low-frequency rumble from road / HVAC / microphone stand
      > 2000 Hz : microphone self-noise, air turbulence, electrical hiss

    Filtering these out before extracting features means the model sees
    only the acoustically meaningful part of the signal — dramatically
    reducing false positives caused by environmental noise.

    Uses a 4th-order Butterworth filter in second-order sections (sos)
    form for numerical stability.
    """
    nyq  = sr / 2.0
    low  = max(low_hz  / nyq, 0.001)                  # normalize frequency
    high = min(high_hz / nyq, 0.999)

    if low >= high:                                    # safety guard
        return audio

    try:
        sos      = butter(N=4, Wn=[low, high], btype="band", output="sos")
        filtered = sosfilt(sos, audio)
        return filtered.astype(np.float32)
    except Exception:
        return audio                                   # if filter fails, use original


def smooth_features(features, window=SMOOTHING_WIN):
    """
    CHANGED (Task 5) — Moving-average smoothing on a 1-D feature array.

    Short spikes in mel-band energy can occur when a single audio frame
    happens to have a transient (door slam, phone vibration, cough near
    microphone).  Smoothing averages out these spikes so that one noisy
    frame does not inflate the anomaly score.

    Uses np.convolve with mode="same" so the output length equals the input.
    """
    if window <= 1 or len(features) < window:
        return features
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(features, kernel, mode="same").astype(np.float32)


# ────────────────────────────────────────────────────────────
# 1.  LOAD AUDIO
# ────────────────────────────────────────────────────────────
def load_audio(file_path):
    """
    Load an audio file, resample to SAMPLE_RATE, convert to mono,
    and apply normalization.

    CHANGED: now calls normalize_audio() after loading so that ALL
    audio — both training samples and test recordings — go through
    the same normalization pipeline.  This is critical: features must
    be computed on normalized audio during both training AND inference.
    """
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    audio     = normalize_audio(audio)                 # CHANGED — normalize on load
    return audio, sr


# ────────────────────────────────────────────────────────────
# 2.  INDIVIDUAL FEATURE EXTRACTORS
# ────────────────────────────────────────────────────────────
def extract_mfcc_features(audio, sr):
    """
    MFCC = Mel-Frequency Cepstral Coefficients.
    Returns mean + std of each coefficient -> shape (N_MFCC*2,)
    """
    mfcc      = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc,  axis=1)
    return mfcc_mean, mfcc_std


def extract_spectral_features(audio, sr):
    """
    Spectral features — brightness and shape of frequency content.
    Returns a list of 4 scalar values.
    """
    spectral_centroid  = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    spectral_rolloff   = np.mean(librosa.feature.spectral_rolloff(y=audio,  sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    return [spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate]


def extract_chroma_features(audio, sr):
    """
    Chroma features — pitch-class energy across 12 bins.
    Returns mean of 12 chroma bins -> shape (12,)
    """
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return np.mean(chroma, axis=1)


def extract_rms_energy(audio):
    """
    Root Mean Square energy — mean + std of loudness over time.
    Returns a list of 2 scalar values.
    """
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    return [np.mean(rms), np.std(rms)]


def extract_mel_spectrogram_features(audio, sr):
    """
    Mean mel-spectrogram energy per band -> shape (N_MELS,)

    CHANGED (Task 5): smoothing is now applied to the per-band mean
    before returning.  This removes short-duration spikes (one or two
    bad frames) that would otherwise make a healthy engine look abnormal.
    """
    mel    = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS,
        hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db      = librosa.power_to_db(mel, ref=np.max)
    mel_mean    = np.mean(mel_db, axis=1)              # (N_MELS,) — mean over time
    mel_smooth  = smooth_features(mel_mean)             # CHANGED — apply smoothing
    return mel_smooth


# ────────────────────────────────────────────────────────────
# 3.  COMBINED FEATURE VECTOR (used by training & prediction)
# ────────────────────────────────────────────────────────────
def extract_features(file_path):
    """
    Master function: load audio -> bandpass filter -> extract ALL features
    -> return one flat feature vector of shape (226,).

    CHANGED (Task 4): a bandpass filter (500-2000 Hz) is now applied to
    the audio BEFORE any feature extraction.  The visualization functions
    (get_spectrogram_data, get_mel_spectrogram_data) are NOT filtered so
    engineers can still see the full-spectrum plots.

    Feature layout (226 total):
        mfcc_mean     : 40
        mfcc_std      : 40
        spectral      :  4  (centroid, bandwidth, rolloff, zcr)
        chroma        : 12
        rms           :  2  (mean, std)
        mel_mean      : 128  (smoothed)
    """
    audio, sr = load_audio(file_path)

    # CHANGED — apply bandpass filter to focus on engine frequencies
    audio_filtered = apply_bandpass_filter(audio, sr)  # CHANGED

    # Extract features on the FILTERED audio
    mfcc_mean, mfcc_std = extract_mfcc_features(audio_filtered, sr)   # CHANGED
    spectral            = extract_spectral_features(audio_filtered, sr)  # CHANGED
    chroma              = extract_chroma_features(audio_filtered, sr)   # CHANGED
    rms                 = extract_rms_energy(audio_filtered)            # CHANGED
    mel_mean            = extract_mel_spectrogram_features(audio_filtered, sr)  # CHANGED

    features = np.concatenate([
        mfcc_mean,   # 40
        mfcc_std,    # 40
        spectral,    #  4
        chroma,      # 12
        rms,         #  2
        mel_mean     # 128
    ])               # Total: 226
    return features


# ────────────────────────────────────────────────────────────
# 4.  VISUALISATION DATA (NOT filtered — shows full spectrum)
# ────────────────────────────────────────────────────────────
def get_spectrogram_data(audio, sr):
    """Full-spectrum dB spectrogram for plotting. Shape: (freq_bins, time_frames)"""
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    D    = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return D


def get_mel_spectrogram_data(audio, sr):
    """Full-spectrum dB mel-spectrogram for plotting. Shape: (N_MELS, time_frames)"""
    mel    = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS,
        hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db