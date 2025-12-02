import numpy as np
from numba import njit
from typing import Dict, Tuple, Literal

# import parselmouth


# ---------- shape helpers ----------
def _ensure_time_major(x: np.ndarray) -> np.ndarray:
    """
    Ensure x is [T, M] (time-major). Accept [M, T] and transpose.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got {x.shape}")
    T, M = x.shape
    # Heuristic: mel bins usually <= 512; frames usually > bins
    if T < M:
        return x.T.astype(np.float32, copy=False)
    return x.astype(np.float32, copy=False)

# ---------- distances (per-frame) ----------
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
                   metric: str = "cosine",
                   window: int | None = None,
                   return_aligned: bool = True):
    """
    Align two mel spectrograms with DTW.

    Parameters
    ----------
    mel_a, mel_b : np.ndarray
        Mel spectrograms in [T,M] or [M,T]. Will be converted to time-major [T,M].
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
    A = _ensure_time_major(mel_a)
    B = _ensure_time_major(mel_b)

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

# ---------- Example ----------
# if __name__ == "__main__":
#     rng = np.random.default_rng(0)
#     # Fake mels: [T,M] = [180 frames, 80 mels]
#     A = rng.normal(0, 1, size=(180, 80)).astype(np.float32)
#     # Create a time-warped version of A for testing
#     idx = np.round(np.linspace(0, 179, 160)).astype(int)
#     B = A[idx] + 0.05 * rng.normal(0, 1, size=(160, 80)).astype(np.float32)

#     cost, path, A_al, B_al = dtw_align_mels(A, B, metric="cosine", window=20, return_aligned=True)
#     print(f"DTW cost: {cost:.3f}, aligned length: {len(path)}")


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

def _dtw_align_indices_1d(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DTW path between 1D series using z-scored, NaN-interpolated copies
    (for stability), then return index arrays ai, bi into the ORIGINAL series.
    """
    # Prepare features for alignment (but we index original signals with the path)
    A_feat = _zscore_1d(_nan_interp_1d(a))[:, None]  # [T,1]
    B_feat = _zscore_1d(_nan_interp_1d(b))[:, None]  # [U,1]

    # Use your numba DTW (time-major already) to get the path
    # metric='l2' is fine for 1D standardized features
    total_cost, path = _dtw_path_numba(A_feat, B_feat, metric=0, window=-1)  # returns (cost, path[L,2])
    ai = path[:, 0].astype(np.int64)
    bi = path[:, 1].astype(np.int64)
    return ai, bi

def _framewise_rfft_power(mel_BxT: np.ndarray, center=True, hann=True) -> np.ndarray:
    """
    Compute rFFT power P(q, m)=|C(q,m)|^2 along mel bins for each frame m.
    mel_BxT: [B, T]
    Returns P of shape [Q, T], with Q = floor(B/2)+1.
    """
    B, T = mel_BxT.shape
    X = mel_BxT.astype(np.float32, copy=False)

    if center:
        X = X - np.mean(X, axis=0, keepdims=True)  # remove DC across bins
    if hann:
        w = np.hanning(B).astype(np.float32)
        X = X * w[:, None]

    C = np.fft.rfft(X, axis=0)         # [Q, T], complex
    P = (C.real**2 + C.imag**2)        # magnitude^2
    return P

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

def hqer_from_power(P_qT: np.ndarray, 
                    q_c: int | None = None,
                    reduction: Literal['none', 'mean', 'median'] = 'none',
                    ) -> float:
    """
    High-Quefrency Energy Ratio per utterance (median over frames).
    P_qT: [Q, T] power. We exclude q=0 from denominators.
    q_c: cutoff index (inclusive) for 'high' band. Default: floor(0.25*Q).
    """
    Q, T = P_qT.shape
    if q_c is None:
        q_c = int(np.floor(0.25 * Q))
        q_c = max(1, min(q_c, Q-1))

    denom = np.sum(P_qT[1:Q, :], axis=0) + 1e-12
    numer = np.sum(P_qT[q_c:Q, :], axis=0)
    per_frame = numer / denom
    
    return _reduce_series(per_frame, reduction=reduction)

def slope_from_power(P_qT: np.ndarray, 
                     q1: int = 1, 
                     q2: int | None = None, 
                     eps: float = 1e-8,
                     reduction: Literal['none', 'mean', 'median'] = 'none',
                     ) -> float:
    """
    Linear slope of log-power vs quefrency (median over frames).
    More negative slope => more smoothing.
    """
    Q, T = P_qT.shape
    if q2 is None:
        q2 = Q - 1
    q = np.arange(q1, q2 + 1, dtype=np.float32)  # [K]
    if q.size < 2:
        return float("nan")

    logP = 10*np.log10(P_qT[q1:q2+1, :] + eps)        # [K, T]
    # Least-squares slope for each frame using polyfit of degree 1
    # y = a*q + b => slope a
    # vectorized: compute per-frame slope
    q_mean = np.mean(q)
    q_var = np.mean((q - q_mean)**2) + 1e-12
    y_mean = np.mean(logP, axis=0)
    cov = np.mean((q[:, None] - q_mean) * (logP - y_mean), axis=0)
    slopes = cov / q_var
    
    return _reduce_series(slopes, reduction=reduction)

def centroid_from_power(P_qT: np.ndarray,
                        reduction: Literal['none', 'mean', 'median'] = 'none',
                        ) -> float:
    """
    Energy-weighted mean quefrency normalized to [0,1] (median over frames).
    Lower => energy concentrated at low q (smoother).
    """
    Q, T = P_qT.shape
    q = np.arange(Q, dtype=np.float32)  # [0..Q-1]
    denom = np.sum(P_qT[1:Q, :], axis=0) + 1e-12  # exclude q=0
    num = np.sum((q[1:Q, None] * P_qT[1:Q, :]), axis=0)
    mean_q = num / denom  # [T]
    return _reduce_series(mean_q, reduction=reduction)

def rolloff_from_power(P_qT: np.ndarray, p: float = 0.95,
                       reduction: Literal['none', 'mean', 'median'] = 'none',
                       ) -> float:
    """
    p (default: 95%) cumulative-energy cutoff quefrency (median over frames).
    Returns q95 in absolute bins (0..Q-1). We ignore q=0 in the cumulative.
    """
    Q, T = P_qT.shape
    P = P_qT.copy()
    P[0, :] = 0.0
    cum = np.cumsum(P, axis=0)                      # [Q, T]
    tot = cum[-1, :] + 1e-12
    target = p * tot
    # For each frame, find smallest q with cum(q) >= target
    # Build a mask and argmax the first True
    ge = cum >= target[None, :]
    # If a column has no True (all zeros), default to q=1
    idx = np.where(np.any(ge, axis=0), np.argmax(ge, axis=0), 1)
    return _reduce_series(idx, reduction=reduction)

def compute_mel_over_smoothing_metrics(mel: np.ndarray,
                                       assume_BxT: bool | None = True,
                                       center: bool = True,
                                       hann: bool = True,
                                       q_c: int | None = None,
                                       reduction: Literal['none', 'mean', 'median'] = 'none',
                                       ) -> dict:
    """
    Compute HQER, Slope, Sharpness, q95 for one utterance.
    mel: 2D array [B, T] or [T, B].
    assume_BxT: if None, auto-detect; else True forces [B,T], False forces [T,B].
    center: subtract per-frame mean across bins before rFFT.
    hann: apply Hann window across bins before rFFT.
    q_c: cutoff for HQER; default = floor(0.25*Q).
    """
    if assume_BxT is True:
        mel_BxT = mel
    elif assume_BxT is False:
        mel_BxT = mel.T 

    P_qT = _framewise_rfft_power(mel_BxT, center=center, hann=hann)

    return {
        "HQER": 100*hqer_from_power(P_qT, q_c=q_c, reduction=reduction),
        "CSlope": slope_from_power(P_qT, reduction=reduction),
        "CCentroid": centroid_from_power(P_qT, reduction=reduction),
        "CRoll95": rolloff_from_power(P_qT, p=0.95, reduction=reduction),
        "Q": int(P_qT.shape[0])
    }
    
def aligned_distance(series_pred, series_ref):
    ai, bi = _dtw_align_indices_1d(series_pred, series_ref)
    series_pred_al, series_ref_al = series_pred[ai], series_ref[bi]
    mae = np.mean(np.abs(series_pred_al - series_ref_al))
    
    return float(mae)

def over_smoothing_metric_aligned(mel_spec_pred, mel_spec_ref, center = True):
    
    scores_pred = compute_mel_over_smoothing_metrics(mel_spec_pred, assume_BxT=True, center=center)
    scores_ref = compute_mel_over_smoothing_metrics(mel_spec_ref, assume_BxT=True, center=center)

    metric_dict = {}
    for k in scores_pred.keys():
        series_pred, series_ref = scores_pred[k], scores_ref[k]
        if not isinstance(series_pred, np.ndarray): continue
        series_mae = aligned_distance(series_pred, series_ref)
        delta_u = _median_ignore_nan(series_pred) - _median_ignore_nan(series_ref)
        metric_dict[f'mae_{k}'] = series_mae 
        metric_dict[f'delta_u_{k}'] = delta_u 
    
    return metric_dict