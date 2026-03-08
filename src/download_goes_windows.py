from __future__ import annotations

import calendar
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from scipy.signal import find_peaks


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "events.csv"
OUTPUT_DIR = BASE_DIR / "data" / "output"
CACHE_DIR = BASE_DIR / "data" / "cache"

DEFAULT_WINDOW_HOURS = 24
DEFAULT_CHANNELS = ["Hp", "Bt"]

TIMEOUT = 60

GOESR_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"
LEGACY_BASE = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "goes-event-review-batch/1.0"})


# ============================================================
# PICKER SETTINGS (current v4-ish baseline)
# ============================================================

PICKER_SETTINGS = {
    "min_peak_prominence": 2.0,
    "min_peak_distance_minutes": 10,
    "double_peak_merge_minutes": 10,
    "double_peak_shallow_dip_nt": 3.5,
    "min_separation_between_g_minutes": 45,
    "search_after_peak_min_minutes": 2,
    "search_after_peak_max_minutes": 30,
    "first_turn_window_minutes": 8,
    "confirm_window_minutes": 15,
    "min_future_drop_8m": 1.2,
    "min_future_drop_15m": 2.4,
    "min_negative_fraction_8m": 0.60,
    "min_negative_fraction_15m": 0.60,
    "max_up_reversal_8m": 1.0,
    "macro_window_minutes": 360,
    "macro_top_quantile": 0.72,
    "macro_flat_slope_abs_threshold": 0.015,
    "spike_jump_nt": 12.0,
}


# ============================================================
# INPUT PARSING
# ============================================================

def parse_user_timestamp(value: str) -> datetime:
    text = str(value).strip()
    formats = [
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%y %H:%M",
        "%m/%d/%y %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Could not parse datetime: {value!r}")


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)
    required = {"event_id", "center_time_utc", "window_hours", "channels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"events.csv is missing required columns: {sorted(missing)}")

    df["center_time_utc"] = df["center_time_utc"].apply(parse_user_timestamp)
    df["window_hours"] = df["window_hours"].fillna(DEFAULT_WINDOW_HOURS).astype(int)
    df["channels"] = df["channels"].fillna("|".join(DEFAULT_CHANNELS)).astype(str)
    return df


# ============================================================
# SATELLITE MAPPING
# ============================================================

def east_satellite_for_time(ts: datetime) -> str:
    if ts < datetime(2003, 4, 1, tzinfo=timezone.utc):
        return "goes08"
    if ts < datetime(2010, 4, 14, tzinfo=timezone.utc):
        return "goes12"
    if ts < datetime(2017, 12, 18, tzinfo=timezone.utc):
        return "goes13"
    if ts < datetime(2025, 4, 7, tzinfo=timezone.utc):
        return "goes16"
    return "goes19"


def is_goesr_satellite(sat: str) -> bool:
    return sat in {"goes16", "goes17", "goes18", "goes19"}


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def download_file(url: str, outpath: Path) -> bool:
    print(f"Downloading: {url}")
    r = SESSION.get(url, timeout=TIMEOUT)
    if r.status_code == 200:
        outpath.write_bytes(r.content)
        return True
    print(f"  -> HTTP {r.status_code}", file=sys.stderr)
    return False


def fetch_text(url: str) -> str | None:
    print(f"Listing: {url}")
    r = SESSION.get(url, timeout=TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.text


# ============================================================
# MODERN GOES-R/U
# ============================================================

def goesr_month_dir(sat: str, day: datetime) -> str:
    yyyy = day.strftime("%Y")
    mm = day.strftime("%m")
    return f"{GOESR_BASE}/{sat}/l2/data/magn-l2-avg1m/{yyyy}/{mm}/"


def find_goesr_filename(sat: str, day: datetime) -> str | None:
    sat_num = sat.replace("goes", "")
    yyyymmdd = day.strftime("%Y%m%d")
    month_url = goesr_month_dir(sat, day)
    html = fetch_text(month_url)

    if html is None:
        return None

    pattern = rf"dn_magn-l2-avg1m_g{sat_num}_d{yyyymmdd}_v[\d\-]+\.nc"
    matches = re.findall(pattern, html)

    if not matches:
        return None

    return sorted(set(matches))[-1]


def ensure_goesr_daily_file(sat: str, day: datetime) -> Path | None:
    tag = day.strftime("%Y%m%d")
    cached = CACHE_DIR / f"{sat}_{tag}.nc"
    if cached.exists():
        print(f"Using cache: {cached.name}")
        return cached

    filename = find_goesr_filename(sat, day)
    if filename is None:
        return None

    url = goesr_month_dir(sat, day) + filename
    if not download_file(url, cached):
        return None
    return cached


# ============================================================
# LEGACY GOES 8-15
# ============================================================

def month_start(day: datetime) -> datetime:
    return day.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_month(day: datetime) -> datetime:
    if day.month == 12:
        return day.replace(year=day.year + 1, month=1, day=1)
    return day.replace(month=day.month + 1, day=1)


def month_range(start: datetime, end: datetime) -> Iterable[datetime]:
    cur = month_start(start)
    end_m = month_start(end)
    while cur <= end_m:
        yield cur
        cur = next_month(cur)


def legacy_month_dir(sat: str, month_day: datetime) -> str:
    yyyy = month_day.strftime("%Y")
    mm = month_day.strftime("%m")
    return f"{LEGACY_BASE}/{yyyy}/{mm}/{sat}/netcdf/"


def legacy_month_filename(sat: str, month_day: datetime) -> str:
    sat_num = sat.replace("goes", "")
    yyyy = month_day.strftime("%Y")
    mm = month_day.strftime("%m")
    last_day = calendar.monthrange(month_day.year, month_day.month)[1]
    return f"g{sat_num}_magneto_1m_{yyyy}{mm}01_{yyyy}{mm}{last_day:02d}.nc"


def ensure_legacy_month_file(sat: str, month_day: datetime) -> Path | None:
    tag = month_day.strftime("%Y%m")
    cached = CACHE_DIR / f"{sat}_{tag}.nc"
    if cached.exists():
        print(f"Using cache: {cached.name}")
        return cached

    filename = legacy_month_filename(sat, month_day)
    url = legacy_month_dir(sat, month_day) + filename
    if not download_file(url, cached):
        return None
    return cached


# ============================================================
# NETCDF HANDLING
# ============================================================

def find_time_column(df: pd.DataFrame) -> str:
    preferred = ["time", "time_tag", "time_utc", "datetime"]
    lower_map = {c.lower(): c for c in df.columns}

    for p in preferred:
        if p in lower_map:
            return lower_map[p]

    for c in df.columns:
        if "time" in c.lower():
            return c

    raise KeyError(f"No time column found. Columns: {list(df.columns)}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        low = col.lower()

        if low in {"hp", "h_p"}:
            rename_map[col] = "Hp"
        elif low in {"he", "h_e"}:
            rename_map[col] = "He"
        elif low in {"hn", "h_n"}:
            rename_map[col] = "Hn"
        elif low in {"bt", "ht", "b_total", "total_field"}:
            rename_map[col] = "Bt"

    return df.rename(columns=rename_map)


def open_netcdf_as_dataframe(nc_path: Path) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    df = ds.to_dataframe().reset_index()
    df = normalize_columns(df)

    time_col = find_time_column(df)
    df["time_utc"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"]).copy()

    keep_cols = ["time_utc"] + [c for c in ["Hp", "He", "Hn", "Bt"] if c in df.columns]
    df = df[keep_cols].copy()
    df = df.sort_values("time_utc").drop_duplicates(subset=["time_utc"])

    return df


# ============================================================
# WINDOW EXTRACTION
# ============================================================

def daterange(start_day: datetime, end_day: datetime) -> Iterable[datetime]:
    cur = start_day
    while cur <= end_day:
        yield cur
        cur += timedelta(days=1)


def extract_window(center_time: datetime, window_hours: int) -> tuple[datetime, datetime]:
    return (
        center_time - timedelta(hours=window_hours),
        center_time + timedelta(hours=window_hours),
    )


def load_event_window(
    event_id: str,
    center_time: datetime,
    window_hours: int,
    channels: List[str],
) -> tuple[pd.DataFrame, str]:
    sat = east_satellite_for_time(center_time)
    start, end = extract_window(center_time, window_hours)

    pieces: List[pd.DataFrame] = []

    if is_goesr_satellite(sat):
        start_day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end_day = end.replace(hour=0, minute=0, second=0, microsecond=0)

        for day in daterange(start_day, end_day):
            local_nc = ensure_goesr_daily_file(sat, day)
            if local_nc is None:
                continue
            pieces.append(open_netcdf_as_dataframe(local_nc))
    else:
        for mon in month_range(start, end):
            local_nc = ensure_legacy_month_file(sat, mon)
            if local_nc is None:
                continue
            pieces.append(open_netcdf_as_dataframe(local_nc))

    if not pieces:
        raise FileNotFoundError(f"No source files available for {event_id}")

    df = pd.concat(pieces, ignore_index=True)
    df = df[(df["time_utc"] >= start) & (df["time_utc"] <= end)].copy()
    df = df.sort_values("time_utc").drop_duplicates(subset=["time_utc"])

    keep_cols = ["time_utc"] + [c for c in channels if c in df.columns]
    df = df[keep_cols].copy()

    return df, sat


# ============================================================
# PICKER
# ============================================================

@dataclass
class DetectionResult:
    g_time_earth_utc: str
    peak_time_earth_utc: str
    score: float
    status: str
    confidence: str
    note: str


def prep_trace(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("time_utc")

    if "Hp" not in out.columns:
        raise ValueError(f"Expected Hp column. Columns found: {list(out.columns)}")

    keep_cols = ["time_utc"] + [c for c in ["Hp", "He", "Hn", "Bt"] if c in out.columns]
    out = out[keep_cols].copy()

    for col in ["Hp", "He", "Hn", "Bt"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Hp"]).copy()

    out = (
        out.set_index("time_utc")
        .resample("1min")
        .median(numeric_only=True)
        .interpolate(limit=5)
        .reset_index()
    )

    # spike flag
    out["jump"] = out["Hp"].diff().abs()
    out["hp_smooth_3"] = out["Hp"].rolling(3, center=True, min_periods=1).median()
    out["hp_baseline_180"] = out["hp_smooth_3"].rolling(180, center=True, min_periods=30).median()
    out["hp_resid"] = out["hp_smooth_3"] - out["hp_baseline_180"]
    out["d1"] = out["hp_smooth_3"].diff()

    macro_w = int(PICKER_SETTINGS["macro_window_minutes"])
    out["hp_macro"] = out["hp_smooth_3"].rolling(macro_w, center=True, min_periods=max(60, macro_w // 4)).median()
    out["hp_macro"] = out["hp_macro"].interpolate(limit=30)
    out["macro_d1"] = out["hp_macro"].diff()
    out["macro_roll_min"] = out["hp_macro"].rolling(macro_w, center=True, min_periods=max(60, macro_w // 4)).min()
    out["macro_roll_max"] = out["hp_macro"].rolling(macro_w, center=True, min_periods=max(60, macro_w // 4)).max()
    rng = out["macro_roll_max"] - out["macro_roll_min"]
    out["macro_pos"] = np.where(rng > 0, (out["hp_macro"] - out["macro_roll_min"]) / rng, np.nan)

    return out


def trough_depth_between(work: pd.DataFrame, i1: int, i2: int) -> float:
    if i2 <= i1 + 1:
        return 0.0
    y1 = float(work.loc[i1, "hp_smooth_3"])
    y2 = float(work.loc[i2, "hp_smooth_3"])
    trough = float(work.loc[i1:i2, "hp_smooth_3"].min())
    return min(y1, y2) - trough


def cluster_peaks(work: pd.DataFrame, peak_idx: np.ndarray) -> List[List[int]]:
    if len(peak_idx) == 0:
        return []

    clusters: List[List[int]] = []
    current = [int(peak_idx[0])]

    for raw_idx in peak_idx[1:]:
        idx = int(raw_idx)
        prev = current[-1]
        dt = idx - prev
        trough_depth = trough_depth_between(work, prev, idx)

        should_merge = (
            dt <= int(PICKER_SETTINGS["double_peak_merge_minutes"])
            and trough_depth < float(PICKER_SETTINGS["double_peak_shallow_dip_nt"])
        )

        if should_merge:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]

    clusters.append(current)
    return clusters


def is_macro_top_region(work: pd.DataFrame, idx: int) -> bool:
    pos = work.loc[idx, "macro_pos"]
    slope = work.loc[idx, "macro_d1"]

    if pd.isna(pos) or pd.isna(slope):
        return False

    near_top = pos >= float(PICKER_SETTINGS["macro_top_quantile"])
    flat_or_turning = abs(float(slope)) <= float(PICKER_SETTINGS["macro_flat_slope_abs_threshold"])

    return bool(near_top and flat_or_turning)


def is_obvious_spike(work: pd.DataFrame, idx: int) -> bool:
    spike_jump = float(PICKER_SETTINGS["spike_jump_nt"])
    here = float(work.loc[idx, "jump"]) if idx < len(work) else 0.0
    prev = float(work.loc[idx - 1, "jump"]) if idx - 1 >= 0 else 0.0

    # Also catches one-minute blips
    return max(here, prev) >= spike_jump


def score_to_confidence(score: float) -> str:
    if score >= 11.0:
        return "high"
    if score >= 7.5:
        return "medium"
    return "low"


def detect_g_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, List[DetectionResult]]:
    work = prep_trace(df).reset_index(drop=True)
    y = work["hp_resid"].to_numpy()

    if len(y) < 120:
        return work, []

    peak_idx, _ = find_peaks(
        y,
        prominence=float(PICKER_SETTINGS["min_peak_prominence"]),
        distance=int(PICKER_SETTINGS["min_peak_distance_minutes"]),
    )

    if len(peak_idx) == 0:
        return work, []

    clusters = cluster_peaks(work, peak_idx)
    final_peaks = [cluster[-1] for cluster in clusters]
    detections: List[DetectionResult] = []

    for peak in final_peaks:
        if is_obvious_spike(work, peak):
            continue

        if is_macro_top_region(work, peak):
            continue

        start_search = peak + int(PICKER_SETTINGS["search_after_peak_min_minutes"])
        end_search = min(
            peak + int(PICKER_SETTINGS["search_after_peak_max_minutes"]),
            len(work) - (int(PICKER_SETTINGS["confirm_window_minutes"]) + 1),
        )
        if start_search >= end_search:
            continue

        chosen_idx: Optional[int] = None
        chosen_score = -np.inf

        for i in range(start_search, end_search):
            if is_macro_top_region(work, i):
                continue

            if is_obvious_spike(work, i):
                continue

            w1 = int(PICKER_SETTINGS["first_turn_window_minutes"])
            w2 = int(PICKER_SETTINGS["confirm_window_minutes"])

            future_8 = float(work.loc[i + w1, "hp_smooth_3"] - work.loc[i, "hp_smooth_3"])
            future_15 = float(work.loc[i + w2, "hp_smooth_3"] - work.loc[i, "hp_smooth_3"])
            neg_frac_8 = float((work.loc[i + 1:i + w1, "d1"] < 0).mean())
            neg_frac_15 = float((work.loc[i + 1:i + w2, "d1"] < 0).mean())
            local_reversal = float(work.loc[i + 1:i + w1, "hp_smooth_3"].max() - work.loc[i, "hp_smooth_3"])
            minutes_after_peak = i - peak

            valid = (
                future_8 <= -float(PICKER_SETTINGS["min_future_drop_8m"])
                and future_15 <= -float(PICKER_SETTINGS["min_future_drop_15m"])
                and neg_frac_8 >= float(PICKER_SETTINGS["min_negative_fraction_8m"])
                and neg_frac_15 >= float(PICKER_SETTINGS["min_negative_fraction_15m"])
                and local_reversal <= float(PICKER_SETTINGS["max_up_reversal_8m"])
            )

            if not valid:
                continue

            # pick first credible downslope
            score = (
                (-future_8) * 1.8
                + (-future_15) * 1.0
                + neg_frac_8 * 4.0
                + neg_frac_15 * 2.0
                - minutes_after_peak * 0.7
            )

            chosen_idx = i
            chosen_score = float(score)
            break

        if chosen_idx is not None:
            status = "provisional" if (chosen_idx - peak) <= 10 else "confirmed"
            confidence = score_to_confidence(chosen_score)

            detections.append(
                DetectionResult(
                    g_time_earth_utc=work.loc[chosen_idx, "time_utc"].isoformat(),
                    peak_time_earth_utc=work.loc[peak, "time_utc"].isoformat(),
                    score=chosen_score,
                    status=status,
                    confidence=confidence,
                    note="first credible downslope after final release peak",
                )
            )

    detections = sorted(detections, key=lambda d: d.g_time_earth_utc)

    filtered: List[DetectionResult] = []
    min_sep = int(PICKER_SETTINGS["min_separation_between_g_minutes"])

    for det in detections:
        if not filtered:
            filtered.append(det)
            continue

        prev_time = pd.Timestamp(filtered[-1].g_time_earth_utc)
        this_time = pd.Timestamp(det.g_time_earth_utc)
        dt_minutes = (this_time - prev_time).total_seconds() / 60.0

        if dt_minutes < min_sep:
            if det.score > filtered[-1].score:
                filtered[-1] = det
        else:
            filtered.append(det)

    return work, filtered


# ============================================================
# OUTPUTS
# ============================================================

def make_event_plot(trace_df: pd.DataFrame, detections: List[DetectionResult], title: str, outpath: Path) -> None:
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(111)

    ax.plot(trace_df["time_utc"], trace_df["Hp"], label="Hp raw", linewidth=0.8)
    ax.plot(trace_df["time_utc"], trace_df["hp_smooth_3"], label="Hp smooth", linewidth=1.5)

    ymax = float(trace_df["hp_smooth_3"].max())
    ymin = float(trace_df["hp_smooth_3"].min())
    yspan = ymax - ymin if ymax != ymin else 1.0
    ytext = ymax - 0.03 * yspan

    for det in detections:
        t = pd.Timestamp(det.g_time_earth_utc)
        ax.axvline(t, linewidth=1.2)
        ax.text(
            t,
            ytext,
            f"{t.strftime('%m-%d %H:%M')}\n{det.status}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )

    ax.set_title(title)
    ax.set_xlabel("UTC")
    ax.set_ylabel("Hp (nT)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ============================================================
# REVIEW PACKET BUILDER
# ============================================================

def write_review_template(outpath: Path, event_id: str, detections: List[DetectionResult]) -> None:
    rows = []
    for det in detections:
        rows.append(
            {
                "event_id": event_id,
                "time_utc": det.g_time_earth_utc,
                "review_status": "",
                "confidence": "",
                "notes": "",
            }
        )

    # Leave a few blank rows so user can add missed picks
    for _ in range(6):
        rows.append(
            {
                "event_id": event_id,
                "time_utc": "",
                "review_status": "",
                "confidence": "",
                "notes": "",
            }
        )

    pd.DataFrame(rows).to_csv(outpath, index=False)


def process_event(row: pd.Series) -> dict:
    event_id = str(row["event_id"])
    center_time = row["center_time_utc"]
    window_hours = int(row["window_hours"])
    channels = [c.strip() for c in str(row["channels"]).split("|") if c.strip()]

    event_dir = OUTPUT_DIR / event_id
    event_dir.mkdir(parents=True, exist_ok=True)

    df, sat = load_event_window(event_id, center_time, window_hours, channels)

    trace_df, detections = detect_g_candidates(df)

    # raw window with metadata columns
    out_df = df.copy()
    out_df["event_id"] = event_id
    out_df["center_time_utc"] = center_time.isoformat()
    out_df["source_satellite"] = sat
    out_df.to_csv(event_dir / f"{event_id}_goes_window.csv", index=False)

    # candidates
    cand_df = pd.DataFrame([asdict(d) for d in detections])
    cand_df.to_csv(event_dir / f"{event_id}_g_candidates.csv", index=False)

    # cleaned trace used by picker
    trace_df.to_csv(event_dir / f"{event_id}_trace_used.csv", index=False)

    # plot
    title = f"{event_id} | GOES-East historical review | center={center_time.strftime('%Y-%m-%d %H:%M UTC')}"
    make_event_plot(trace_df, detections, title, event_dir / f"{event_id}_plot.png")

    # review template
    write_review_template(event_dir / f"{event_id}_review_template.csv", event_id, detections)

    # summary
    summary = {
        "event_id": event_id,
        "center_time_utc": center_time.isoformat(),
        "window_hours_each_side": window_hours,
        "source_satellite": sat,
        "n_candidates": len(detections),
        "candidate_times_utc": [d.g_time_earth_utc for d in detections],
    }
    with open(event_dir / f"{event_id}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> int:
    print(f"Reading events from: {INPUT_FILE}")
    events = load_events(INPUT_FILE)

    created = 0
    failed = 0
    summaries = []
    failures = []

    for _, row in events.iterrows():
        event_id = str(row["event_id"])
        print(f"\n===== Processing {event_id} =====")
        try:
            summary = process_event(row)
            summaries.append(summary)
            created += 1
        except Exception as exc:
            failed += 1
            failures.append(
                {
                    "event_id": event_id,
                    "center_time_utc": row["center_time_utc"].isoformat(),
                    "error": str(exc),
                }
            )
            print(f"[FAIL] {event_id}: {exc}", file=sys.stderr)

    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "batch_summary.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(OUTPUT_DIR / "batch_failures.csv", index=False)

    print("\nFinished.")
    print(f"  created: {created}")
    print(f"  failed:  {failed}")

    # succeed if at least one event succeeded
    return 0 if created > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
