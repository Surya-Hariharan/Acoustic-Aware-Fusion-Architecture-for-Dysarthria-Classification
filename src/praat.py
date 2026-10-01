"""
Framewise Praat (parselmouth) features for the segmental and suprasegmental
branches, on the same 10 ms frame grid as the MFCCs.

  extract_segmental_extra_sequence  F1, F2, F3 (Hz) and HNR (dB) — appended to
                                    MFCC+delta+delta-delta to form the 43-channel
                                    segmental input (speech-focused profile).
  extract_suprasegmental_sequence   F0 (semitones re 100 Hz), a binary voicing
                                    mask, intensity (dB) — the 3-channel
                                    suprasegmental input (temporal-preserving
                                    profile).

Both analyse only the real-audio prefix waveform[:valid_length], then zero-pad
to total_frames. Neither ever raises: a Praat failure leaves zeros, so a
difficult clip contributes no signal from that branch rather than aborting a
batch. These run once per utterance when the feature store is built, never
during training.

The four constants below are part of src.feature_store.store_signature();
changing one invalidates every stored chunk.
"""

from typing import Dict

import numpy as np
import parselmouth
from parselmouth.praat import call

PITCH_FLOOR = 75.0          # Hz — adult speech
PITCH_CEILING = 600.0       # Hz — generous for female/pathological voices
HNR_SILENCE_FLOOR = -100.0  # dB — Praat parks silent/unvoiced frames near -200
FRAME_HOP_SECONDS = 0.01    # = MEL_KWARGS["hop_length"] (160) at 16 kHz


def _hz_to_semitones(f0_hz: np.ndarray, reference_hz: float = 100.0) -> np.ndarray:
    """Semitones re 100 Hz: equal steps are equal perceived pitch changes
    regardless of a speaker's register, unlike linear Hz."""
    return 12.0 * np.log2(np.clip(f0_hz, 1e-3, None) / reference_hz)


def _valid_frames(valid_length: int, total_frames: int) -> int:
    from src.preprocessing import mfcc_frame_count
    return min(mfcc_frame_count(valid_length), total_frames) if valid_length > 0 else 0


def extract_suprasegmental_sequence(waveform: np.ndarray, sr: int, valid_length: int,
                                    total_frames: int) -> Dict[str, np.ndarray]:
    """(total_frames,) float32 arrays "f0_semitones", "voicing", "intensity_db".

    Unvoiced frames carry F0 = 0 with voicing = 0: no contour is interpolated
    through them, so the model is shown where pitch was genuinely absent."""
    f0 = np.zeros(total_frames, dtype=np.float32)
    voicing = np.zeros(total_frames, dtype=np.float32)
    intensity = np.zeros(total_frames, dtype=np.float32)
    valid_frames = _valid_frames(valid_length, total_frames)
    if valid_frames <= 1:
        return {"f0_semitones": f0, "voicing": voicing, "intensity_db": intensity}

    try:
        sound = parselmouth.Sound(np.asarray(waveform[:valid_length], dtype=np.float64),
                                  sampling_frequency=sr)
        pitch = sound.to_pitch(time_step=FRAME_HOP_SECONDS,
                               pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        intensity_obj = sound.to_intensity(minimum_pitch=PITCH_FLOOR, time_step=FRAME_HOP_SECONDS)

        raw_f0 = pitch.selected_array["frequency"]
        voiced = raw_f0 > 0
        raw_semitones = np.zeros_like(raw_f0, dtype=np.float32)
        raw_semitones[voiced] = _hz_to_semitones(raw_f0[voiced])

        times = pitch.ts()
        raw_intensity = np.zeros(len(times), dtype=np.float32)
        for i, t in enumerate(times):
            try:
                value = call(intensity_obj, "Get value at time", t, "Linear")
                raw_intensity[i] = value if value is not None and not np.isnan(value) else 0.0
            except Exception:
                pass

        n = min(valid_frames, len(raw_semitones), len(raw_intensity))
        f0[:n] = raw_semitones[:n]
        voicing[:n] = voiced[:n].astype(np.float32)
        intensity[:n] = raw_intensity[:n]
    except Exception:
        pass
    return {"f0_semitones": f0, "voicing": voicing, "intensity_db": intensity}


def extract_segmental_extra_sequence(waveform: np.ndarray, sr: int, valid_length: int,
                                     total_frames: int) -> Dict[str, np.ndarray]:
    """(total_frames,) float32 arrays "f1_hz", "f2_hz", "f3_hz", "hnr_db"
    (Burg formants, cross-correlation harmonicity)."""
    f1 = np.zeros(total_frames, dtype=np.float32)
    f2 = np.zeros(total_frames, dtype=np.float32)
    f3 = np.zeros(total_frames, dtype=np.float32)
    hnr = np.zeros(total_frames, dtype=np.float32)
    valid_frames = _valid_frames(valid_length, total_frames)
    if valid_frames <= 1:
        return {"f1_hz": f1, "f2_hz": f2, "f3_hz": f3, "hnr_db": hnr}

    try:
        sound = parselmouth.Sound(np.asarray(waveform[:valid_length], dtype=np.float64),
                                  sampling_frequency=sr)
        formant = sound.to_formant_burg(time_step=FRAME_HOP_SECONDS, max_number_of_formants=5,
                                        maximum_formant=5500, window_length=0.025,
                                        pre_emphasis_from=50)
        harmonicity = call(sound, "To Harmonicity (cc)", FRAME_HOP_SECONDS, PITCH_FLOOR, 0.1, 1.0)

        for i in range(valid_frames):
            t = i * FRAME_HOP_SECONDS
            for formant_idx, arr in ((1, f1), (2, f2), (3, f3)):
                try:
                    value = call(formant, "Get value at time", formant_idx, t, "Hertz", "Linear")
                    arr[i] = value if value is not None and not np.isnan(value) else 0.0
                except Exception:
                    pass
            try:
                h = call(harmonicity, "Get value at time", t, "Linear")
                hnr[i] = h if (h is not None and not np.isnan(h) and h > HNR_SILENCE_FLOOR) else 0.0
            except Exception:
                pass
    except Exception:
        pass
    return {"f1_hz": f1, "f2_hz": f2, "f3_hz": f3, "hnr_db": hnr}
