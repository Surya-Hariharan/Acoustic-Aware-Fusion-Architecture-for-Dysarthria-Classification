"""
Speech-processing EDA: short-time time/frequency-domain parameters,
wideband/narrowband spectrograms, cepstral analysis, Linear Prediction
analysis, and a combined VAD + GAD (glottal/voicing activity) three-way
silence/unvoiced/voiced segmentation.

Deliberately separate from src/visualization.py (which houses the existing
Praat-statistics/VAD-validation EDA) — this module is the "classical
speech-processing technique" panel set, called from
notebooks/02_feature_analysis.ipynb's speech-processing-EDA and feature-
extraction-summary stages. Every function returns raw arrays as well as
optionally plotting, so the same computation can feed a notebook table
without recomputation.

All plots use src.style's shared color system (apply_style, SEQUENTIAL_CMAP,
VOICING_COLORS, FORMANT_COLORS) so every heatmap/segmentation panel in the
EDA notebook is visually consistent with the rest of the project's figures.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from src import config
from src.style import apply_style, SEQUENTIAL_CMAP, VOICING_COLORS, FORMANT_COLORS

apply_style()

# ---------------------------------------------------------------------------
# Frame geometry shared by every short-time function below. 25ms/10ms is the
# textbook default (matches src.preprocessing's own MFCC frame hop of 10ms /
# n_fft=400 = 25ms at 16kHz), so these panels sit on the same time axis as
# the MFCC/formant/F0 panels elsewhere in the notebook.
# ---------------------------------------------------------------------------
FRAME_MS = 25
HOP_MS = 10


def _frame_signal(waveform: np.ndarray, sr: int, frame_ms: int = FRAME_MS,
                   hop_ms: int = HOP_MS) -> Tuple[np.ndarray, np.ndarray]:
    """1-D waveform -> (frames, frame_times_s). frames: (n_frames, frame_len)."""
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    n_frames = max(1, 1 + (len(waveform) - frame_len) // hop_len) if len(waveform) >= frame_len else 1
    frames = np.zeros((n_frames, frame_len), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        chunk = waveform[start:start + frame_len]
        frames[i, :len(chunk)] = chunk
    times = (np.arange(n_frames) * hop_len + frame_len / 2) / sr
    return frames, times


# ---------------------------------------------------------------------------
# Short-time TIME-DOMAIN parameters
# ---------------------------------------------------------------------------
def short_time_energy(waveform: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """(frame_times_s, energy) — mean squared amplitude per frame. High in
    voiced segments (strong periodic excitation), low in silence."""
    frames, times = _frame_signal(waveform, sr)
    energy = (frames ** 2).mean(axis=1)
    return times, energy


def zero_crossing_rate(waveform: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """(frame_times_s, zcr) — sign-change rate per frame, normalized to
    [0, 1] of the frame length. High for unvoiced/fricative noise (broadband,
    no dominant periodicity), low for voiced segments (energy concentrated
    at F0 and its harmonics)."""
    frames, times = _frame_signal(waveform, sr)
    signs = np.sign(frames)
    signs[signs == 0] = 1
    crossings = np.abs(np.diff(signs, axis=1)) > 0
    zcr = crossings.mean(axis=1)
    return times, zcr


# ---------------------------------------------------------------------------
# Short-time FREQUENCY-DOMAIN parameters
# ---------------------------------------------------------------------------
def spectral_centroid_rolloff(waveform: np.ndarray, sr: int, rolloff_ratio: float = 0.85
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frame_times_s, centroid_hz, rolloff_hz) per frame.

    centroid: the spectrum's "center of mass" — low for voiced (energy
    concentrated in low harmonics), high for fricatives/sibilants.
    rolloff: the frequency below which `rolloff_ratio` of the spectral
    energy is contained — a robust proxy for where energy "runs out",
    complementary to the centroid.
    """
    frames, times = _frame_signal(waveform, sr)
    window = np.hanning(frames.shape[1])
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sr)

    power = spectrum ** 2
    total = power.sum(axis=1, keepdims=True)
    total_safe = np.clip(total, 1e-12, None)
    centroid = (power * freqs[None, :]).sum(axis=1) / total_safe[:, 0]

    cumulative = np.cumsum(power, axis=1) / total_safe
    rolloff = np.array([
        freqs[np.searchsorted(cumulative[i], rolloff_ratio)]
        if total[i, 0] > 1e-12 else 0.0
        for i in range(frames.shape[0])
    ])
    return times, centroid, rolloff


# ---------------------------------------------------------------------------
# Spectrograms — wideband (short window, good time / poor frequency
# resolution: individual glottal pulses visible as vertical striations,
# harmonics blurred together) vs narrowband (long window, poor time / good
# frequency resolution: individual F0 harmonics resolved as horizontal
# bands, pulse-level timing lost). The classic textbook analysis-window
# trade-off, computed identically except for window length.
# ---------------------------------------------------------------------------
def compute_spectrogram(waveform: np.ndarray, sr: int, window_ms: float
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(freqs_hz, times_s, magnitude_db) via torch.stft with a Hann window of
    the given length; hop fixed at 1/4 of the window (standard STFT overlap)."""
    win_len = max(16, int(sr * window_ms / 1000))
    hop_len = max(1, win_len // 4)
    n_fft = 1
    while n_fft < win_len:
        n_fft *= 2
    wav_t = torch.from_numpy(waveform.astype(np.float32))
    window = torch.hann_window(win_len)
    stft = torch.stft(wav_t, n_fft=n_fft, hop_length=hop_len, win_length=win_len,
                       window=window, return_complex=True, center=True)
    magnitude_db = 20 * torch.log10(stft.abs().clamp(min=1e-8))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = np.arange(stft.shape[1]) * hop_len / sr
    return freqs, times, magnitude_db.numpy()


def plot_wideband_narrowband_spectrogram(waveform: np.ndarray, sr: int,
                                          wideband_ms: float = 5.0,
                                          narrowband_ms: float = 30.0,
                                          title: str = "", show: bool = False,
                                          out_name: str = "eda_wideband_vs_narrowband_spectrogram.png") -> str:
    """Side-by-side wideband vs narrowband spectrogram of the same utterance,
    same colormap/scale so the resolution trade-off is visually direct."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    vmax = None
    for ax, window_ms, label in ((axes[0], wideband_ms, f"Wideband ({wideband_ms:.0f} ms window)"),
                                  (axes[1], narrowband_ms, f"Narrowband ({narrowband_ms:.0f} ms window)")):
        freqs, times, mag_db = compute_spectrogram(waveform, sr, window_ms)
        if vmax is None:
            vmax = mag_db.max()
        im = ax.pcolormesh(times, freqs, mag_db, cmap=SEQUENTIAL_CMAP,
                            vmin=vmax - 80, vmax=vmax, shading="auto")
        ax.set_ylim(0, min(8000, sr / 2))
        ax.set_xlabel("Time (s)")
        ax.set_title(label)
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axes, label="Magnitude (dB)", fraction=0.03, pad=0.02)
    fig.suptitle(title or "Wideband vs. narrowband spectrogram", y=1.02)
    return _save(fig, out_name, show)


# ---------------------------------------------------------------------------
# Cepstral analysis — the real cepstrum of one voiced frame, which the
# textbook motivation for MFCC comes from directly: the log-spectrum is
# (approximately) the sum of a slowly-varying formant envelope and a
# fast-varying glottal-pulse-train component; the inverse FFT of the log-
# spectrum ("cepstrum") separates them by quefrency — a sharp peak at the
# pitch period's quefrency (low-order harmonics repeat there), and a
# low-quefrency region that IS the formant envelope MFCC keeps.
# ---------------------------------------------------------------------------
def real_cepstrum(frame: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """(quefrency_s, cepstrum) for one frame, via ifft(log|fft|)."""
    window = np.hanning(len(frame))
    spectrum = np.fft.fft(frame * window)
    log_magnitude = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.ifft(log_magnitude).real
    quefrency = np.arange(len(frame)) / sr
    return quefrency, cepstrum


def plot_cepstral_analysis(waveform: np.ndarray, sr: int, frame_center_s: float,
                            frame_ms: float = 32.0, pitch_range_hz: Tuple[float, float] = (60, 400),
                            title: str = "", show: bool = False,
                            out_name: str = "eda_cepstral_analysis.png") -> str:
    """One voiced frame's waveform, log-magnitude spectrum, and real cepstrum
    (with the plausible pitch-period quefrency band shaded) — three panels
    that motivate MFCC's log/quefrency split before the MFCC panel itself."""
    frame_len = int(sr * frame_ms / 1000)
    center = int(frame_center_s * sr)
    frame = waveform[max(0, center - frame_len // 2): center + frame_len // 2]
    if len(frame) < frame_len:
        frame = np.pad(frame, (0, frame_len - len(frame)))

    window = np.hanning(len(frame))
    spectrum = np.fft.rfft(frame * window)
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)
    log_mag = 20 * np.log10(np.abs(spectrum) + 1e-8)
    quefrency, cepstrum = real_cepstrum(frame, sr)

    q_lo, q_hi = 1.0 / pitch_range_hz[1], 1.0 / pitch_range_hz[0]
    n_half = len(quefrency) // 2
    peak_idx = np.argmax(cepstrum[int(q_lo * sr):min(int(q_hi * sr), n_half)]) + int(q_lo * sr)
    f0_estimate = 1.0 / quefrency[peak_idx] if quefrency[peak_idx] > 0 else float("nan")

    fig, axes = plt.subplots(3, 1, figsize=(7, 8))
    axes[0].plot(np.arange(len(frame)) / sr * 1000, frame, color="#4c72b0", linewidth=0.8)
    axes[0].set_title("Windowed frame")
    axes[0].set_xlabel("Time (ms)")

    axes[1].plot(freqs, log_mag, color="#dd8452", linewidth=0.8)
    axes[1].set_title("Log-magnitude spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_xlim(0, min(5000, sr / 2))

    axes[2].plot(quefrency[:n_half] * 1000, cepstrum[:n_half], color="#55a868", linewidth=0.8)
    axes[2].axvspan(q_lo * 1000, q_hi * 1000, color="#c44e52", alpha=0.15,
                     label=f"plausible pitch quefrency ({pitch_range_hz[0]:.0f}-{pitch_range_hz[1]:.0f} Hz)")
    axes[2].axvline(quefrency[peak_idx] * 1000, color="#c44e52", linestyle="--",
                     label=f"peak -> F0 ~= {f0_estimate:.0f} Hz")
    axes[2].set_title("Real cepstrum")
    axes[2].set_xlabel("Quefrency (ms)")
    axes[2].legend(fontsize=8)

    fig.suptitle(title or "Cepstral analysis", y=1.0)
    fig.tight_layout()
    return _save(fig, out_name, show)


# ---------------------------------------------------------------------------
# Linear Prediction (LPC) analysis — the all-pole vocal-tract model
# complementary to the FFT-based spectral envelope: an order-p LPC fits
# x[n] ~= sum_k a_k x[n-k], and 1 / |1 - sum a_k z^-k| traces a smooth
# envelope through the FFT spectrum's harmonic peaks (formants), without
# needing cepstral liftering.
# ---------------------------------------------------------------------------
def _levinson_durbin(autocorr: np.ndarray, order: int) -> np.ndarray:
    """Autocorrelation -> LPC coefficients a[1..order] (a[0] omitted, implicit
    1.0), via the standard recursive Levinson-Durbin solution to the normal
    equations — avoids forming/inverting the (order x order) Toeplitz matrix
    directly."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = autocorr[0]
    if e <= 0:
        return a
    for i in range(1, order + 1):
        acc = autocorr[i] + np.dot(a[1:i], autocorr[i - 1:0:-1])
        k = -acc / e
        a_new = a.copy()
        a_new[i] = k
        a_new[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a = a_new
        e *= (1 - k ** 2)
        if e <= 0:
            break
    return a


def lpc_envelope(frame: np.ndarray, sr: int, order: int = 16, n_fft: int = 1024
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(freqs_hz, fft_magnitude_db, lpc_envelope_db) for one frame — the FFT
    spectrum and the order-`order` LPC all-pole envelope fit through it,
    on the same axis for direct comparison."""
    window = np.hanning(len(frame))
    windowed = frame * window
    spectrum = np.fft.rfft(windowed, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fft_db = 20 * np.log10(np.abs(spectrum) + 1e-8)

    autocorr = np.correlate(windowed, windowed, mode="full")[len(windowed) - 1:]
    autocorr = autocorr[:order + 1]
    a = _levinson_durbin(autocorr, order)

    w, h = np.linspace(0, np.pi, n_fft // 2 + 1), None
    z = np.exp(-1j * np.outer(w, np.arange(order + 1)))
    denom = z @ a
    lpc_mag = 1.0 / np.clip(np.abs(denom), 1e-8, None)
    gain = np.sqrt(max(np.dot(a, autocorr[:order + 1]), 1e-12))
    lpc_db = 20 * np.log10(lpc_mag * gain + 1e-8)
    return freqs, fft_db, lpc_db


def plot_lpc_envelope(waveform: np.ndarray, sr: int, frame_center_s: float,
                       order: int = 16, frame_ms: float = 32.0,
                       title: str = "", show: bool = False,
                       out_name: str = "eda_lpc_envelope.png") -> str:
    """FFT spectrum with the LPC all-pole envelope overlaid, for one
    (presumed voiced) frame — the Linear Prediction Analysis panel."""
    frame_len = int(sr * frame_ms / 1000)
    center = int(frame_center_s * sr)
    frame = waveform[max(0, center - frame_len // 2): center + frame_len // 2]
    if len(frame) < frame_len:
        frame = np.pad(frame, (0, frame_len - len(frame)))

    freqs, fft_db, lpc_db = lpc_envelope(frame, sr, order=order)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freqs, fft_db, color="#8c8c8c", linewidth=0.7, label="FFT spectrum")
    ax.plot(freqs, lpc_db, color="#c44e52", linewidth=1.8, label=f"LPC envelope (order {order})")
    ax.set_xlim(0, min(5000, sr / 2))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title or f"Linear Prediction analysis (order {order})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, out_name, show)


# ---------------------------------------------------------------------------
# VAD (speech vs. silence) + GAD (glottal/voicing activity, frame-level) ->
# one combined three-way segmentation: silence / unvoiced / voiced.
# GAD here is the same voicing decision src.praat.extract_suprasegmental_
# sequence already computes for the Suprasegmental branch (Praat's pitch/
# point-process voicing logic) — reused, not recomputed, so the EDA panel
# and the model's actual input agree by construction.
# ---------------------------------------------------------------------------
def three_way_segmentation(total_frames: int, frame_hop_s: float,
                            speech_start_s: float, speech_end_s: float,
                            voicing_mask: np.ndarray) -> np.ndarray:
    """Per-frame labels in {0: silence, 1: unvoiced, 2: voiced}. `voicing_mask`
    is expected on the SAME frame grid as `total_frames` (see
    src.praat.extract_suprasegmental_sequence's `voicing` output) — frames
    outside the VAD speech span are silence regardless of the voicing mask
    (Praat's pitch tracker can return spurious voiced frames in background
    noise; VAD's speech/non-speech decision takes precedence there)."""
    frame_times = np.arange(total_frames) * frame_hop_s
    in_speech = (frame_times >= speech_start_s) & (frame_times <= speech_end_s)
    labels = np.zeros(total_frames, dtype=np.int64)
    labels[in_speech & (voicing_mask > 0.5)] = 2
    labels[in_speech & (voicing_mask <= 0.5)] = 1
    return labels


def plot_voiced_unvoiced_silence(waveform: np.ndarray, sr: int, labels: np.ndarray,
                                  frame_hop_s: float, title: str = "",
                                  ste: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                  zcr: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                  show: bool = False,
                                  out_name: str = "eda_voiced_unvoiced_silence.png") -> str:
    """Waveform with silence/unvoiced/voiced regions shaded (VOICING_COLORS),
    plus short-time energy and ZCR beneath if provided — the panel that
    visually justifies the segmentation (voiced = high energy + low ZCR,
    unvoiced = lower energy + high ZCR, silence = near-zero energy)."""
    n_panels = 1 + (ste is not None) + (zcr is not None)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 2.4 * n_panels), sharex=True)
    axes = np.atleast_1d(axes)

    t_wave = np.arange(len(waveform)) / sr
    axes[0].plot(t_wave, waveform, color="0.25", linewidth=0.5)
    _shade_regions(axes[0], labels, frame_hop_s)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(title or "Silence / unvoiced / voiced segmentation")

    panel = 1
    if ste is not None:
        axes[panel].plot(ste[0], ste[1], color="#4c72b0", linewidth=1.0)
        _shade_regions(axes[panel], labels, frame_hop_s)
        axes[panel].set_ylabel("Short-time energy")
        panel += 1
    if zcr is not None:
        axes[panel].plot(zcr[0], zcr[1], color="#dd8452", linewidth=1.0)
        _shade_regions(axes[panel], labels, frame_hop_s)
        axes[panel].set_ylabel("Zero-crossing rate")
        panel += 1

    axes[-1].set_xlabel("Time (s)")
    handles = [plt.matplotlib.patches.Patch(facecolor=c, alpha=0.35, label=name.capitalize())
               for name, c in VOICING_COLORS.items()]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=8, bbox_to_anchor=(0.98, 1.02))
    fig.tight_layout()
    return _save(fig, out_name, show)


def _shade_regions(ax, labels: np.ndarray, frame_hop_s: float) -> None:
    names = {0: "silence", 1: "unvoiced", 2: "voiced"}
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            ax.axvspan(start * frame_hop_s, i * frame_hop_s,
                       color=VOICING_COLORS[names[labels[start]]], alpha=0.35, linewidth=0)
            start = i


# ---------------------------------------------------------------------------
def _save(fig, filename: str, show: bool) -> str:
    out_path = config.SIGNAL_FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out_path)
