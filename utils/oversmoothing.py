"""
Cepstral-domain oversmoothing metrics for mel-spectrogram-based TTS models.

This module implements a set of objective metrics designed to quantify
spectral oversmoothing in predicted mel-spectrograms. The metrics operate
in the cepstral (quefrency) domain by applying an rFFT across mel-frequency
bins on a frame-wise basis and analyzing the resulting power distribution.

The implemented measures include:
- High-Quefrency Energy Ratio (HQER),
- Cepstral Slope (CSlope),
- Cepstral Centroid (CCentroid),
- Cepstral Rolloff (CRoll95),

which jointly capture changes in fine spectral detail, spectral contrast,
and energy distribution that are not well reflected by conventional
reconstruction losses such as L1/L2 distance or spectral convergence.

All metrics can be computed frame-wise, aggregated at the utterance level,
and compared between predicted and reference signals using DTW alignment.
They are intended for evaluation and analysis, but are differentiable (or
smoothly approximable) and may also be used as training objectives.

The definitions and motivation for these metrics are described in:

    L. Nippert,
    "Arabic TTS with FastPitch: Reproducible Baselines, Adversarial Training,
     and Oversmoothing Analysis", arXiv:2512.00937, 2025.

See https://arxiv.org/abs/2512.00937 for details.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from typing import Dict, Tuple, Literal, Optional, Union

# ---------- Distances (per-frame) ----------
@njit(cache=True)
def _l2_row(a: np.ndarray, b: np.ndarray) -> float:
    s = 0.0
    for k in range(a.shape[0]):
        d = a[k] - b[k]
        s += d * d
    return np.sqrt(s)

@njit(cache=True)
def _cosine_row(a: np.ndarray, b: np.ndarray) -> float:
    num = 0.0
    na = 0.0
    nb = 0.0
    for k in range(a.shape[0]):
        ak = a[k]
        bk = b[k]
        num += ak * bk
        na += ak * ak
        nb += bk * bk
    den = np.sqrt(na) * np.sqrt(nb) + 1e-12
    sim = num / den
    # Clamp numeric drift
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    # Cosine distance
    return 1.0 - sim

# ---------- Helpers ----------
def _median_ignore_nan(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")

def _mean_ignore_nan(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")

def _reduce_series(x: np.ndarray, 
                   reduction: Literal['none', 'mean', 'median'] = 'none',
                   ) -> np.ndarray:
    if reduction == 'mean':
        return _mean_ignore_nan(x)
    elif reduction == 'median':
        return _median_ignore_nan(x)
    else:
        return x

def _nan_interp_1d(x: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs; if all-NaN, return zeros."""
    x = x.astype(np.float32, copy=True)
    n = x.size
    nan = np.isnan(x)
    if not np.any(nan):
        return x
    if np.all(nan):
        return np.zeros_like(x)
    idx = np.arange(n, dtype=np.float32)
    x[nan] = np.interp(idx[nan], idx[~nan], x[~nan])
    return x

def _zscore_1d(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0.0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - m) / s).astype(np.float32)


# ---------- DTW core ----------
@njit(cache=True)
def _dtw_path_numba(A: np.ndarray,
                    B: np.ndarray,
                    metric: int = 0,
                    window: int = -1) -> tuple:
    """
    Compute DTW between time-major mel specs A [Ta,M], B [Tb,M].
    metric: 0=L2, 1=cosine
    window: Sakoe-Chiba band radius (in frames); -1 disables the band.
    Returns (total_cost, path) where path is int32 array [L,2] of (i,j) indices.
    """
    Ta, M = A.shape
    Tb = B.shape[0]

    inf = np.float32(1e30)
    # Accumulated cost matrix (+1 padding for easy boundaries)
    D = np.empty((Ta + 1, Tb + 1), dtype=np.float32)
    D[:] = inf
    D[0, 0] = 0.0

    # Backpointer matrix: 0=up (i-1,j), 1=left (i,j-1), 2=diag (i-1,j-1), -1=unreachable
    P = np.full((Ta, Tb), -1, dtype=np.int8)

    use_band = window >= 0
    w = window if window >= 0 else 0

    for i in range(1, Ta + 1):
        # band limits for j (1-indexed in D)
        j_min = 1
        j_max = Tb
        if use_band:
            j_min = max(1, i - w)
            j_max = min(Tb, i + w)

        # prefetch row vector
        ai = A[i - 1]

        for j in range(j_min, j_max + 1):
            # local frame distance
            if metric == 0:
                cost = _l2_row(ai, B[j - 1])
            else:
                cost = _cosine_row(ai, B[j - 1])

            # choose predecessor with min accumulated cost
            up = D[i - 1, j]
            left = D[i, j - 1]
            diag = D[i - 1, j - 1]

            # argmin among (up, left, diag)
            best = up
            bp = 0  # up
            if left < best:
                best = left
                bp = 1  # left
            if diag < best:
                best = diag
                bp = 2  # diag

            D[i, j] = cost + best
            P[i - 1, j - 1] = bp

    # backtrack
    i = Ta - 1
    j = Tb - 1
    # Worst-case path length is Ta+Tb
    path = np.empty((Ta + Tb, 2), dtype=np.int32)
    L = 0
    while i >= 0 and j >= 0:
        path[L, 0] = i
        path[L, 1] = j
        bp = P[i, j]
        if bp == 2:
            i -= 1
            j -= 1
        elif bp == 0:
            i -= 1
        elif bp == 1:
            j -= 1
        else:
            # Unreachable cell (shouldn’t happen if band wasn’t too tight)
            break
        L += 1

    # reverse path to ascending time
    out = np.empty((L, 2), dtype=np.int32)
    for k in range(L):
        out[k, 0] = path[L - 1 - k, 0]
        out[k, 1] = path[L - 1 - k, 1]

    total_cost = float(D[Ta, Tb])
    return total_cost, out

def dtw_align_mels(mel_a: np.ndarray,
                   mel_b: np.ndarray,
                   metric: Literal["cosine", "l2"] = "cosine",
                   window: Optional[int] = None,
                   return_aligned: bool = True
                   ):
    """
    Align two mel spectrograms with DTW.

    Parameters
    ----------
    mel_a, mel_b : np.ndarray
        Mel spectrograms in [B,T].
    metric : {"cosine","l2"}
        Frame distance.
    window : int or None
        Sakoe-Chiba band radius (frames). None disables the band.
    return_aligned : bool
        If True, also return time-warped aligned copies (A', B') by path sampling.

    Returns
    -------
    total_cost : float
    path : np.ndarray of shape [L,2]
        Warping path as (i,j) index pairs into the time axis of mel_a/mel_b.
    (A_aligned, B_aligned) : np.ndarray, np.ndarray  (only if return_aligned=True)
        Time-aligned sequences [L, M] built by following the path.
    """
    A = mel_a.T
    B = mel_b.T

    mcode = 0 if metric.lower() == "l2" else 1
    w = -1 if window is None else int(window)

    total_cost, path = _dtw_path_numba(A, B, metric=mcode, window=w)

    if not return_aligned:
        return total_cost, path

    # Build aligned sequences by following the path (gather rows)
    L = path.shape[0]
    M = A.shape[1]
    A_al = np.empty((L, M), dtype=np.float32)
    B_al = np.empty((L, M), dtype=np.float32)
    for k in range(L):
        A_al[k, :] = A[path[k, 0], :]
        B_al[k, :] = B[path[k, 1], :]
    return total_cost, path, A_al, B_al

def _dtw_align_indices_1d(
    a: np.ndarray, 
    b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DTW path between 1D series using z-scored, NaN-interpolated copies
    (for stability), then return index arrays ai, bi into the ORIGINAL series.
    """
    # Prepare features for alignment (but we index original signals with the path)
    A_feat = _zscore_1d(_nan_interp_1d(a))[:, None]  # [T,1]
    B_feat = _zscore_1d(_nan_interp_1d(b))[:, None]  # [U,1]

    # Use numba DTW (time-major already) to get the path
    _, path = _dtw_path_numba(A_feat, B_feat, metric=0, window=-1)  # returns (cost, path[L,2])
    ai = path[:, 0].astype(np.int64)
    bi = path[:, 1].astype(np.int64)
    return ai, bi

def aligned_mae_distance(series_pred: np.ndarray, 
                         series_ref: np.ndarray
                         ) -> float:
    """
    Compute the mean absolute error (MAE) between two 1D series after DTW alignment.

    Parameters
    ----------
    series_pred : np.ndarray
        Predicted time series (e.g., per-frame metric values).
    series_ref : np.ndarray
        Reference time series to align against.

    Returns
    -------
    mae : float
        Mean absolute error between the DTW-aligned series.

    Notes
    -----
    Dynamic time warping (DTW) is used to compensate for temporal misalignment
    between prediction and reference. This is particularly important for
    frame-wise spectral or prosodic metrics in TTS, where small timing
    deviations should not dominate the error.
    """
    # Obtain DTW alignment indices for both series
    ai, bi = _dtw_align_indices_1d(series_pred, series_ref)

    # Align both sequences
    series_pred_al = series_pred[ai]
    series_ref_al = series_ref[bi]

    # Compute MAE on aligned sequences
    mae = np.mean(np.abs(series_pred_al - series_ref_al))
    return float(mae)

# ---------- Oversmoothing metrics ----------

def framewise_rfft_power(mel_BxT: np.ndarray, 
                         center: bool = True, 
                         hann: bool = True) -> np.ndarray:
    """
    Compute frame-wise rFFT power of mel-spectrograms along the mel-bin axis.

    Parameters
    ----------
    mel_BxT : np.ndarray
        Log-mel or mel-spectrogram of shape [B, T], where B is the number
        of mel bands and T the number of time frames.
    center : bool, optional
        If True, subtract the per-frame mean across mel bins to remove
        the DC component (default: True).
    hann : bool, optional
        If True, apply a Hann window across mel bins before the rFFT
        to reduce boundary artifacts (default: True).

    Returns
    -------
    P_qT : np.ndarray
        Power spectrum |C(q,m)|² of shape [Q, T], where
        Q = floor(B / 2) + 1.

    Notes
    -----
    The rFFT is applied across the mel-frequency axis for each time frame,
    producing a quefrency-domain representation. High quefrency indices
    correspond to fine spectral structure, while low indices reflect
    smooth spectral envelopes.
    """
    B, T = mel_BxT.shape
    X = mel_BxT.astype(np.float32, copy=False)

    if center:
        # Remove per-frame DC offset across mel bins
        X = X - np.mean(X, axis=0, keepdims=True)

    if hann:
        # Apply Hann window across mel bins
        w = np.hanning(B).astype(np.float32)
        X = X * w[:, None]

    # rFFT along mel axis
    C = np.fft.rfft(X, axis=0)      # [Q, T], complex
    P = (C.real**2 + C.imag**2)     # power spectrum

    return P


def hqer_from_power(P_qT: np.ndarray, 
                    q_c: Optional[int] = None,
                    reduction: Literal['none', 'mean', 'median'] = 'none',                 
                    ) -> float:
    """
    Compute the High-Quefrency Energy Ratio (HQER).

    Parameters
    ----------
    P_qT : np.ndarray
        Power spectrum of shape [Q, T] obtained from rFFT.
    q_c : int, optional
        Cutoff quefrency index separating low and high quefrency regions.
        Defaults to floor(0.25 * Q).
    reduction : {'none', 'mean', 'median'}, optional
        Reduction applied across time frames (default: 'none').

    Returns
    -------
    hqer : float or np.ndarray
        High-quefrency energy ratio per frame or reduced over frames.

    Notes
    -----
    HQER measures the proportion of energy located in high quefrency
    coefficients relative to total non-DC energy. Higher values indicate
    richer fine spectral detail, while lower values suggest oversmoothing.
    """
    Q, T = P_qT.shape

    if q_c is None:
        q_c = int(np.floor(0.25 * Q))
        q_c = max(1, min(q_c, Q - 1))

    # Exclude DC component (q=0)
    denom = np.sum(P_qT[1:Q, :], axis=0) + 1e-12
    numer = np.sum(P_qT[q_c:Q, :], axis=0)

    per_frame = numer / denom
    return _reduce_series(per_frame, reduction=reduction)


def slope_from_power(P_qT: np.ndarray, 
                     q1: int = 1, 
                     q2: Optional[int] = None, 
                     eps: float = 1e-8,
                     reduction: Literal['none', 'mean', 'median'] = 'none',
                     ) -> float:
    """
    Compute the linear slope of power (in dB) versus quefrency (cepstral slope: CSlope).

    Parameters
    ----------
    P_qT : np.ndarray
        Power spectrum of shape [Q, T].
    q1 : int, optional
        Lower quefrency bound (default: 1).
    q2 : int, optional
        Upper quefrency bound (default: Q - 1).
    eps : float, optional
        Small constant to avoid log(0).
    reduction : {'none', 'mean', 'median'}, optional
        Reduction applied across time frames.

    Returns
    -------
    slope : float or np.ndarray
        Linear slope value(s).

    Notes
    -----
    A more negative slope indicates faster decay of spectral detail toward
    higher quefrencies, which is characteristic of oversmoothed spectra.
    """
    Q, T = P_qT.shape
    if q2 is None:
        q2 = Q - 1

    q = np.arange(q1, q2 + 1, dtype=np.float32)
    if q.size < 2:
        return float("nan")

    P_db_qT = 10 * np.log10(P_qT[q1:q2 + 1, :] + eps)

    # Vectorized least-squares slope computation
    q_mean = np.mean(q)
    q_var = np.mean((q - q_mean)**2) + 1e-12
    y_mean = np.mean(P_db_qT, axis=0)
    cov = np.mean((q[:, None] - q_mean) * (P_db_qT - y_mean), axis=0)

    slopes = cov / q_var
    return _reduce_series(slopes, reduction=reduction)


def centroid_from_power(P_qT: np.ndarray,
                        reduction: Literal['none', 'mean', 'median'] = 'none',
                        ) -> float:
    """
    Compute the energy-weighted mean quefrency (cepstral centroid: CCentroid).

    Parameters
    ----------
    P_qT : np.ndarray
        Power spectrum of shape [Q, T].
    reduction : {'none', 'mean', 'median'}, optional
        Reduction applied across time frames.

    Returns
    -------
    centroid : float or np.ndarray
        Energy-weighted mean quefrency.

    Notes
    -----
    Lower centroid values indicate energy concentration at low quefrencies
    (smoother spectra), while higher values correspond to increased fine
    spectral structure.
    """
    Q, T = P_qT.shape
    q = np.arange(Q, dtype=np.float32)

    # Ignore DC component
    denom = np.sum(P_qT[1:Q, :], axis=0) + 1e-12
    num = np.sum(q[1:Q, None] * P_qT[1:Q, :], axis=0)

    mean_q = num / denom
    return _reduce_series(mean_q, reduction=reduction)


def rolloff_from_power(P_qT: np.ndarray, 
                       p: float = 0.95,
                       reduction: Literal['none', 'mean', 'median'] = 'none',
                       ) -> float:
    """
    Compute the cumulative-energy quefrency roll-off (cepstral rolloff: CRoll95 by default).

    Parameters
    ----------
    P_qT : np.ndarray
        Power spectrum of shape [Q, T].
    p : float, optional
        Cumulative energy threshold (default: 0.95).
    reduction : {'none', 'mean', 'median'}, optional
        Reduction applied across time frames.

    Returns
    -------
    q_roll : float or np.ndarray
        Quefrency index at which p fraction of energy is reached.

    Notes
    -----
    Higher roll-off values indicate that significant energy extends toward
    higher quefrencies, corresponding to richer fine spectral detail.
    """
    P = P_qT.copy()
    P[0, :] = 0.0  # ignore DC

    cum = np.cumsum(P, axis=0)
    tot = cum[-1, :] + 1e-12
    target = p * tot

    ge = cum >= target[None, :]
    idx = np.where(np.any(ge, axis=0), np.argmax(ge, axis=0), 1)

    return _reduce_series(idx, reduction=reduction)


def compute_mel_oversmoothing_metrics(
    mel: np.ndarray,                              
    center: bool = True,
    hann: bool = True,
    q_c: Optional[int] = None,
    reduction: Literal['none', 'mean', 'median'] = 'none',
    ) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute cepstral-domain oversmoothing metrics for a single utterance.

    Parameters
    ----------
    mel : np.ndarray
        Mel-spectrogram of shape [B, T].
    center : bool, optional
        Remove per-frame DC component before rFFT.
    hann : bool, optional
        Apply Hann window across mel bins before rFFT.
    q_c : int, optional
        Cutoff quefrency index for HQER computation.
    reduction : {'none', 'mean', 'median'}, optional
        Reduction applied across time frames.

    Returns
    -------
    metrics : dict
        Dictionary containing HQER, CSlope, CCentroid, CRoll95, and Q.

    Notes
    -----
    These metrics quantify different aspects of spectral oversmoothing,
    with higher values generally indicating richer spectral detail.
    
    References
    ----------
    - Arabic TTS with FastPitch: Reproducible Baselines, Adversarial Training, and Oversmoothing Analysis (https://arxiv.org/abs/2512.00937)
    """
    P_qT = framewise_rfft_power(mel, center=center, hann=hann)

    return {
        "HQER": 100 * hqer_from_power(P_qT, q_c=q_c, reduction=reduction),
        "CSlope": slope_from_power(P_qT, reduction=reduction),
        "CCentroid": centroid_from_power(P_qT, reduction=reduction),
        "CRoll95": rolloff_from_power(P_qT, p=0.95, reduction=reduction),
        "Q": int(P_qT.shape[0]),
    }
    

def oversmoothing_metrics_aligned(
    mel_spec_pred: np.ndarray,
    mel_spec_ref: np.ndarray, 
    center: bool = True,
    hann: bool = True
    ) -> Dict[str, float]:
    """
    Compute DTW-aligned oversmoothing errors between predicted and reference mel-spectrograms.

    Parameters
    ----------
    mel_spec_pred : np.ndarray
        Predicted mel-spectrogram [B, T].
    mel_spec_ref : np.ndarray
        Reference mel-spectrogram [B, T].
    center : bool, optional
        Remove per-frame DC component before rFFT.
    hann : bool, optional
        Apply Hann window across mel bins before rFFT.

    Returns
    -------
    metrics : dict
        Dictionary containing frame-wise MAE and utterance-level differences
        for each oversmoothing metric.

    Notes
    -----
    Frame-wise MAE captures local spectral discrepancies after DTW alignment,
    while utterance-level differences quantify systematic bias toward over-
    or under-smoothing.
    
    References
    ----------
    - Arabic TTS with FastPitch: Reproducible Baselines, Adversarial Training, and Oversmoothing Analysis (https://arxiv.org/abs/2512.00937)
    """
    scores_pred = compute_mel_oversmoothing_metrics(
        mel_spec_pred, center=center, hann=hann
    )
    scores_ref = compute_mel_oversmoothing_metrics(
        mel_spec_ref, center=center, hann=hann
    )

    metric_dict = {}
    for k in scores_pred.keys():
        series_pred = scores_pred[k]
        series_ref = scores_ref[k]

        if not isinstance(series_pred, np.ndarray):
            continue

        # Frame-wise aligned MAE
        series_mae = aligned_mae_distance(series_pred, series_ref)

        # Utterance-level median difference
        delta_u = _median_ignore_nan(series_pred) - _median_ignore_nan(series_ref)

        metric_dict[f"mae_{k}"] = series_mae
        metric_dict[f"delta_u_{k}"] = delta_u

    return metric_dict


# ---------- Example ----------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Fake mels: [B,T] = [80 bands, 180 frames]
    mel_a = rng.normal(0, 1, size=(80, 180)).astype(np.float32)
    # Create a time-warped version of A for testing
    idx = np.round(np.linspace(0, 179, 160)).astype(int)
    mel_b = mel_a[:,idx] + 0.05 * rng.normal(0, 1, size=(80, 160)).astype(np.float32)
    print("Shapes of mels:", mel_a.shape, mel_b.shape)

    metrics = compute_mel_oversmoothing_metrics(mel_a, reduction='median')
    print("Metrics", metrics)
    distances = oversmoothing_metrics_aligned(mel_a, mel_b)
    print("Distances", distances)
