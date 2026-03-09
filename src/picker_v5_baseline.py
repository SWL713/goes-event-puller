from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


SETTINGS = {
    "min_peak_prominence": 2.5,
    "min_peak_distance": 12,
    "macro_window": 360,
    "macro_top_cutoff": 0.72,
    "min_future_drop_10": 1.5,
    "min_future_drop_20": 2.8,
    "min_negative_fraction": 0.60,
    "max_spike_jump": 12.0,
    "min_g_separation": 45,
    "candidate_search_start": 2,
    "candidate_search_end": 35,
}


def resample_1min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("time_utc").resample("1min").median(numeric_only=True)
    df = df.interpolate(limit=5)
    return df.reset_index()


def smooth_trace(series: pd.Series) -> pd.Series:
    return series.rolling(3, center=True, min_periods=1).median()


def compute_macro(series: pd.Series, window: int = 360) -> pd.Series:
    macro = series.rolling(window, center=True, min_periods=60).median()
    return macro.interpolate(limit=30)


def prep_trace(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().sort_values("time_utc")

    if "Hp" not in work.columns:
        raise ValueError(f"Expected Hp column. Columns found: {list(work.columns)}")

    work["Hp"] = pd.to_numeric(work["Hp"], errors="coerce")
    work = work.dropna(subset=["Hp"]).copy()

    work = resample_1min(work)

    work["hp_smooth"] = smooth_trace(work["Hp"])
    work["hp_macro"] = compute_macro(work["hp_smooth"], window=SETTINGS["macro_window"])
    work["hp_resid"] = work["hp_smooth"] - work["hp_macro"]
    work["d1"] = work["hp_smooth"].diff()
    work["jump"] = work["Hp"].diff().abs()

    roll_min = work["hp_macro"].rolling(360, min_periods=60).min()
    roll_max = work["hp_macro"].rolling(360, min_periods=60).max()
    denom = roll_max - roll_min
    work["macro_pos"] = np.where(denom > 0, (work["hp_macro"] - roll_min) / denom, np.nan)

    return work


def is_bad_point(work: pd.DataFrame, idx: int) -> bool:
    if idx < 0 or idx >= len(work):
        return True

    if pd.notna(work.loc[idx, "jump"]) and work.loc[idx, "jump"] > SETTINGS["max_spike_jump"]:
        return True

    if pd.notna(work.loc[idx, "macro_pos"]) and work.loc[idx, "macro_pos"] > SETTINGS["macro_top_cutoff"]:
        return True

    return False


def detect_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = prep_trace(df)

    y = work["hp_resid"].values
    peaks, _ = find_peaks(
        y,
        prominence=SETTINGS["min_peak_prominence"],
        distance=SETTINGS["min_peak_distance"],
    )

    candidates = []

    for p in peaks:
        if p >= len(work):
            continue

        if is_bad_point(work, p):
            continue

        start = p + SETTINGS["candidate_search_start"]
        end = min(p + SETTINGS["candidate_search_end"], len(work) - 21)

        chosen = None

        for i in range(start, end):
            if i + 20 >= len(work):
                break

            if is_bad_point(work, i):
                continue

            future10 = work.loc[i + 10, "hp_smooth"] - work.loc[i, "hp_smooth"]
            future20 = work.loc[i + 20, "hp_smooth"] - work.loc[i, "hp_smooth"]
            neg_frac = (work.loc[i + 1:i + 10, "d1"] < 0).mean()
            local_reversal = work.loc[i + 1:i + 8, "hp_smooth"].max() - work.loc[i, "hp_smooth"]

            if (
                future10 < -SETTINGS["min_future_drop_10"]
                and future20 < -SETTINGS["min_future_drop_20"]
                and neg_frac > SETTINGS["min_negative_fraction"]
                and local_reversal < 1.2
            ):
                score = (
                    (-future10) * 1.8
                    + (-future20) * 1.0
                    + neg_frac * 4.0
                    - (i - p) * 0.45
                    + 1.0
                )

                chosen = {
                    "g_time": work.loc[i, "time_utc"],
                    "peak_time": work.loc[p, "time_utc"],
                    "score": float(score),
                    "minutes_after_peak": int(i - p),
                    "path": "post_peak",
                }
                break

        if chosen is not None:
            candidates.append(chosen)

    if not candidates:
        empty = pd.DataFrame(columns=["g_time", "peak_time", "score", "minutes_after_peak", "path"])
        return work, empty

    cand_df = pd.DataFrame(candidates)
    cand_df["g_time"] = pd.to_datetime(cand_df["g_time"], utc=True)
    cand_df["peak_time"] = pd.to_datetime(cand_df["peak_time"], utc=True)
    cand_df = cand_df.sort_values("g_time").reset_index(drop=True)

    filtered_rows = []
    last_keep_time = None
    last_keep_score = None

    for _, row in cand_df.iterrows():
        if last_keep_time is None:
            filtered_rows.append(row)
            last_keep_time = row["g_time"]
            last_keep_score = row["score"]
            continue

        dt_minutes = (row["g_time"] - last_keep_time).total_seconds() / 60.0

        if dt_minutes < SETTINGS["min_g_separation"]:
            if row["score"] > last_keep_score:
                filtered_rows[-1] = row
                last_keep_time = row["g_time"]
                last_keep_score = row["score"]
        else:
            filtered_rows.append(row)
            last_keep_time = row["g_time"]
            last_keep_score = row["score"]

    cand_df = pd.DataFrame(filtered_rows).reset_index(drop=True)

    return work, cand_df
