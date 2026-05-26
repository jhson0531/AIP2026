import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

EPS = 1e-8


def _safe_slope(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return 0.0
    xm, ym = x[mask], values[mask]
    cov = np.cov(xm, ym)[0, 1]
    var = np.var(xm)
    return float(cov / (var + EPS))


def _gini(values: np.ndarray) -> float:
    v = values[~np.isnan(values)]
    v = v[v >= 0]
    if len(v) == 0:
        return 0.0
    v = np.sort(v)
    n = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) / (n * np.sum(v) + EPS)) - (n + 1) / n)


def _burstiness(values: np.ndarray) -> float:
    v = values[~np.isnan(values)]
    if len(v) < 2:
        return 0.0
    mu, sigma = np.mean(v), np.std(v)
    return float((sigma - mu) / (sigma + mu + EPS))


def _autocorr_lag1(values: np.ndarray) -> float:
    v = values[~np.isnan(values)]
    if len(v) < 2:
        return 0.0
    return float(np.corrcoef(v[:-1], v[1:])[0, 1])


def _find_local_peaks(values: np.ndarray) -> list:
    peaks = []
    for i in range(1, len(values) - 1):
        if not np.isnan(values[i]) and values[i] > values[i - 1] and values[i] > values[i + 1]:
            peaks.append(i)
    return peaks


def compute_freq_features(freq_mat: np.ndarray, horizon: int,
                          freq_mask: np.ndarray, active_threshold: float = 1.0) -> pd.DataFrame:
    """Compute frequency features for a given horizon.
    Samples with observed_months < horizon get NaN rows (excluded in experiments).
    """
    N = freq_mat.shape[0]
    prefix = f"h{horizon}_"
    records = []

    for i in range(N):
        obs = freq_mat[i, :horizon]  # use only up to horizon months
        mask = freq_mask[i, :horizon]
        # If fewer than horizon months observed → NaN row
        if mask.sum() < horizon:
            records.append(None)
            continue

        v = obs.copy()
        v[~mask] = np.nan  # should not happen here, but defensive

        f = {}
        # Basic stats
        valid = v[~np.isnan(v)]
        f[f"{prefix}freq_sum"] = float(np.nansum(v))
        f[f"{prefix}freq_mean"] = float(np.nanmean(v)) if len(valid) > 0 else 0.0
        f[f"{prefix}freq_median"] = float(np.nanmedian(v)) if len(valid) > 0 else 0.0
        f[f"{prefix}freq_std"] = float(np.nanstd(v)) if len(valid) > 0 else 0.0
        f[f"{prefix}freq_min"] = float(np.nanmin(v)) if len(valid) > 0 else 0.0
        f[f"{prefix}freq_max"] = float(np.nanmax(v)) if len(valid) > 0 else 0.0
        f[f"{prefix}freq_first"] = float(v[0]) if not np.isnan(v[0]) else 0.0
        f[f"{prefix}freq_last"] = float(v[-1]) if not np.isnan(v[-1]) else 0.0
        f[f"{prefix}active_month_count"] = float(np.sum(v >= active_threshold))
        f[f"{prefix}zero_month_count"] = float(np.sum(v == 0))
        f[f"{prefix}nonzero_month_count"] = float(np.sum(v > 0))

        # Trend
        f[f"{prefix}freq_slope"] = _safe_slope(v)
        log_v = np.log1p(np.where(np.isnan(v), np.nan, np.maximum(v, 0)))
        f[f"{prefix}freq_log_slope"] = _safe_slope(log_v)
        first, last = f[f"{prefix}freq_first"], f[f"{prefix}freq_last"]
        f[f"{prefix}freq_first_to_last_diff"] = last - first
        f[f"{prefix}freq_first_to_last_ratio"] = last / (first + EPS)
        f[f"{prefix}freq_last_to_max_ratio"] = last / (f[f"{prefix}freq_max"] + EPS)

        half = max(horizon // 2, 1)
        early_v = v[:half]
        late_v = v[half:]
        f[f"{prefix}freq_early_mean"] = float(np.nanmean(early_v)) if len(early_v) > 0 else 0.0
        f[f"{prefix}freq_late_mean"] = float(np.nanmean(late_v)) if len(late_v) > 0 else 0.0
        f[f"{prefix}freq_late_to_early_ratio"] = f[f"{prefix}freq_late_mean"] / (f[f"{prefix}freq_early_mean"] + EPS)

        # Peak
        peak_idx = int(np.nanargmax(v))
        f[f"{prefix}peak_value"] = float(np.nanmax(v))
        f[f"{prefix}peak_month"] = float(peak_idx + 1)
        f[f"{prefix}time_to_peak"] = float(peak_idx)

        post_peak = v[peak_idx + 1:] if peak_idx + 1 < len(v) else np.array([])
        f[f"{prefix}post_peak_slope"] = _safe_slope(post_peak) if len(post_peak) >= 2 else 0.0
        log_pp = np.log1p(np.maximum(post_peak, 0))
        f[f"{prefix}post_peak_log_slope"] = _safe_slope(log_pp) if len(log_pp) >= 2 else 0.0
        peak_val = f[f"{prefix}peak_value"]
        f[f"{prefix}post_peak_decay_rate"] = (
            (peak_val - float(np.nanmean(post_peak))) / (peak_val + EPS)
            if len(post_peak) > 0 else 0.0
        )
        f[f"{prefix}peak_sharpness"] = peak_val / (f[f"{prefix}freq_mean"] + EPS)
        local_peaks = _find_local_peaks(v)
        f[f"{prefix}peak_count"] = float(len(local_peaks))

        # Survival
        active = (v >= active_threshold).astype(float)
        # Longest streak
        max_streak, cur_streak = 0, 0
        for a in active:
            if a:
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 0
        f[f"{prefix}longest_active_streak"] = float(max_streak)

        # Current active streak (from end)
        cur_streak = 0
        for a in reversed(active):
            if a:
                cur_streak += 1
            else:
                break
        f[f"{prefix}current_active_streak"] = float(cur_streak)

        f[f"{prefix}retention_ratio"] = f[f"{prefix}freq_last"] / (peak_val + EPS)
        total_sum = f[f"{prefix}freq_sum"] + EPS
        half2 = len(v) // 2
        f[f"{prefix}late_activity_ratio"] = float(np.nansum(v[half2:])) / total_sum

        # Revival: count drops followed by rises after peak
        revival_count = 0
        was_dropping = False
        for j in range(peak_idx + 1, len(v) - 1):
            if not np.isnan(v[j]) and not np.isnan(v[j + 1]):
                if v[j] < v[j - 1]:
                    was_dropping = True
                elif was_dropping and v[j] > v[j - 1]:
                    revival_count += 1
                    was_dropping = False
        f[f"{prefix}revival_count"] = float(revival_count)
        f[f"{prefix}has_revival"] = float(revival_count > 0)

        # Distribution
        nonneg = np.maximum(valid, 0)
        if nonneg.sum() > 0:
            p = nonneg / nonneg.sum()
            f[f"{prefix}freq_entropy"] = float(scipy_entropy(p + EPS))
        else:
            f[f"{prefix}freq_entropy"] = 0.0
        f[f"{prefix}freq_gini"] = _gini(v)
        f[f"{prefix}freq_cv"] = f[f"{prefix}freq_std"] / (f[f"{prefix}freq_mean"] + EPS)
        f[f"{prefix}freq_burstiness"] = _burstiness(v)
        f[f"{prefix}freq_autocorr_lag1"] = _autocorr_lag1(v)

        records.append(f)

    # Build DataFrame, filling None rows with NaN
    all_keys = None
    for r in records:
        if r is not None:
            all_keys = list(r.keys())
            break

    rows = []
    for r in records:
        if r is None:
            rows.append({k: np.nan for k in all_keys})
        else:
            rows.append(r)

    return pd.DataFrame(rows)
