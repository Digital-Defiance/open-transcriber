"""Onset evidence for chart QA (stage 0a plan, D11). All on 16 kHz audio with 10 ms frames.

Three kinds of evidence, any of which can confirm a chart note's onset:

- (a) spectral-flux peaks of the separated vocal: STFT 1024/160 with a Hann window, L = log(1 + 100|X|),
  flux = sum over bins of max(0, L[t] - L[t-1]), normalised by its 99th percentile; a peak is a local maximum
  within ±3 frames that is at least the median of ±25 frames plus 0.10; peaks closer than 30 ms keep the higher.
  Frame t is stamped at its centre, t * 10 ms. Log compression meets the window's leading edge, so an abrupt
  onset peaks about 20 ms early (measured on tone bursts and clicks); the QA's ±100 ms tolerance absorbs it;
- (b) CREPE voicing onsets: a voiced frame after at least 3 unvoiced frames;
- (c) CREPE pitch changes: the median of the next 5 voiced frames differs from the median of the previous 5
  by at least 0.75 semitone (one time per run of such frames, where the change is largest).
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 1024
HOP = 160
FRAME_SECONDS = HOP / SAMPLE_RATE

PEAK_HALF_WIDTH = 3
MEDIAN_HALF_WIDTH = 25
PEAK_DELTA = 0.10
MIN_GAP = 0.030
UNVOICED_BEFORE = 3
PITCH_CONTEXT = 5
PITCH_STEP = 0.75


def _require_16k(rate: int) -> None:
    if rate != SAMPLE_RATE:
        raise ValueError(f"onset evidence needs 16 kHz audio, got {rate} Hz: resample first")


def spectral_flux(audio: np.ndarray, rate: int) -> np.ndarray:
    """Normalised spectral flux per 10 ms frame (frames centred at t * 10 ms, like CREPE's)."""
    _require_16k(rate)
    from scipy.signal import get_window

    window = get_window("hann", N_FFT).astype(np.float64)
    padded = np.pad(np.asarray(audio, dtype=np.float64), (N_FFT // 2, N_FFT // 2))
    total = 1 + len(audio) // HOP
    flux = np.zeros(total)
    previous = None
    for start in range(0, total, 2048):
        stop = min(total, start + 2048)
        idx = HOP * np.arange(start, stop)[:, None] + np.arange(N_FFT)[None, :]
        spectrum = np.log1p(100.0 * np.abs(np.fft.rfft(padded[idx] * window, axis=1)))
        if previous is not None:
            spectrum_with_prev = np.vstack([previous[None, :], spectrum])
        else:
            spectrum_with_prev = np.vstack([spectrum[:1], spectrum])
        diff = np.maximum(0.0, np.diff(spectrum_with_prev, axis=0)).sum(axis=1)
        flux[start:stop] = diff
        previous = spectrum[-1]
    scale = np.percentile(flux, 99)
    return flux / scale if scale > 0 else flux


def flux_peaks(flux: np.ndarray) -> np.ndarray:
    """Peak times in seconds (D11 rule (a))."""
    n = len(flux)
    peaks = []
    for t in range(n):
        lo, hi = max(0, t - PEAK_HALF_WIDTH), min(n, t + PEAK_HALF_WIDTH + 1)
        if flux[t] <= 0 or flux[t] < flux[lo:hi].max():
            continue
        mlo, mhi = max(0, t - MEDIAN_HALF_WIDTH), min(n, t + MEDIAN_HALF_WIDTH + 1)
        if flux[t] >= np.median(flux[mlo:mhi]) + PEAK_DELTA:
            peaks.append(t)
    kept: list[int] = []
    for t in peaks:
        if kept and (t - kept[-1]) * FRAME_SECONDS < MIN_GAP:
            if flux[t] > flux[kept[-1]]:
                kept[-1] = t
            continue
        kept.append(t)
    return np.array(kept, dtype=np.float64) * FRAME_SECONDS


def voicing_onsets(voiced: np.ndarray) -> np.ndarray:
    """Times of voiced frames preceded by at least 3 unvoiced frames (frames before the start count as unvoiced)."""
    voiced = np.asarray(voiced, dtype=bool)
    padded = np.concatenate([np.zeros(UNVOICED_BEFORE, dtype=bool), voiced])
    times = []
    for t in np.flatnonzero(voiced):
        if not padded[t:t + UNVOICED_BEFORE].any():
            times.append(t)
    return np.array(times, dtype=np.float64) * FRAME_SECONDS


def pitch_change_onsets(midi: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """Times where the voiced pitch steps by at least 0.75 semitone (D11 rule (c))."""
    frames = np.flatnonzero(np.asarray(voiced, dtype=bool))
    pitch = np.asarray(midi, dtype=np.float64)[frames]
    k = PITCH_CONTEXT
    change = np.zeros(len(frames))
    for i in range(k, len(frames) - k + 1):
        change[i] = abs(np.median(pitch[i:i + k]) - np.median(pitch[i - k:i]))
    times = []
    i = 0
    while i < len(frames):
        if change[i] >= PITCH_STEP:
            j = i
            while j + 1 < len(frames) and change[j + 1] >= PITCH_STEP and frames[j + 1] == frames[j] + 1:
                j += 1
            # The middle of the frames where the change is largest: a clean step at frame s gives
            # equal changes on s-2..s+2 (5-frame medians), whose middle is s.
            run = change[i:j + 1]
            top = np.flatnonzero(run >= run.max() - 1e-9)
            times.append(frames[i + int(top[len(top) // 2])])
            i = j + 1
        else:
            i += 1
    return np.array(times, dtype=np.float64) * FRAME_SECONDS


def evidence(audio: np.ndarray, rate: int, midi: np.ndarray, voiced: np.ndarray) -> dict[str, np.ndarray]:
    """All three kinds of onset evidence, in seconds."""
    _require_16k(rate)
    return {
        "flux": flux_peaks(spectral_flux(audio, rate)),
        "voicing": voicing_onsets(voiced),
        "pitch": pitch_change_onsets(midi, voiced),
    }


def nearest(times: np.ndarray, target: float) -> float:
    """Signed distance from ``target`` to the nearest time (target - time); inf when there is none."""
    if len(times) == 0:
        return float("inf")
    i = int(np.argmin(np.abs(times - target)))
    return float(target - times[i])
