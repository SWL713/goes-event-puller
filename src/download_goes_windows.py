from __future__ import annotations

import io
import sys
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import xarray as xr


# ----------------------------
# CONFIG
# ----------------------------

DEFAULT_WINDOW_HOURS = 24
DEFAULT_CHANNELS = ["Hp", "Bt"]   # add "He", "Hn" if wanted

# Output folders
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "events.csv"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"

# NOAA / NCEI bases
GOESR_BASE = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access"
# Legacy GOES 1-15 archive is separate. You may need to adjust the exact path once you confirm the file pattern you want.
GOES_LEGACY_BASE = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access"

TIMEOUT = 60


# ----------------------------
# SATELLITE MAPPING
# ----------------------------

def east_satellite_for_time(ts: datetime) -> str:
    """
    Return the GOES-East spacecraft for a UTC datetime.
    Edit here if you want different historical assumptions.

    Current mapping used:
    - GOES-8  : 1998-06-01 to 2003-04-01
    - GOES-12 : 2003-04-01 to 2010-04-14
    - GOES-13 : 2010-04-14 to 2017-12-18
    - GOES-16 : 2017-12-18 to 2025-04-07
    - GOES-19 : 2025-04-07 onward
    """
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


# ----------------------------
# INPUT PARSING
# ----------------------------

def parse_user_timestamp(value: str) -> datetime:
    """
    Parses strings like:
    5/15/05 6:35
    10/10/24 22:32

    Assumes UTC and two-digit year:
    00-69 -> 2000-2069
    70-99 -> 1970-1999
    """
    dt = datetime.strptime(value.strip(), "%m/%d/%y %H:%M")
    return dt.replace(tzinfo=timezone.utc)


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "center_time_utc" not in df.columns:
        raise ValueError("events.csv must include a 'center_time_utc' column.")
    if "event_id" not in df.columns:
        df["event_id"] = [f"event_{i+1:03d}" for i in range(len(df))]
    df["center_time_utc"] = df["center_time_utc"].apply(parse_user_timestamp)
    return df


# ----------------------------
# HTTP HELPERS
# ----------------------------

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "goes-event-puller/1.0"})


def download_file(url: str, outpath: Path) -> bool:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    r = SESSION.get(url, timeout=TIMEOUT)
    if r.status_code == 200:
        outpath.write_bytes(r.content)
        return True
    return False


# ----------------------------
# FILE DISCOVERY / URL BUILDERS
# ----------------------------

def goesr_daily_url(sat: str, day: datetime) -> str:
    """
    Example target style for GOES-R MAG daily avg1m files.
    You may need to tweak the filename pattern if NOAA revises versioning.
    """
    sat_num = sat.replace("goes", "")
    yyyy = day.strftime("%Y")
    mm = day.strftime("%m")
    yyyymmdd = day.strftime("%Y%m%d")

    # Common NCEI naming for GOES-R MAG 1-minute averages:
    # dn_magn-l2-avg1m_g16_d20240510_v2-0-0.nc
    filename = f"dn_magn-l2-avg1m_g{sat_num}_d{yyyymmdd}_v2-0-0.nc"

    return (
        f"{GOESR_BASE}/{sat}/magnetometer-l2-avg1m/{yyyy}/{mm}/{filename}"
    )


def legacy_daily_url_candidates(sat: str, day: datetime) -> list[str]:
    """
    Legacy GOES 1-15 paths have changed over time and can differ from GOES-R.
    This returns a few candidate paths to try.

    You will likely only need to adjust this function once after testing.
    """
    sat_num = sat.replace("goes", "")
    yyyy = day.strftime("%Y")
    mm = day.strftime("%m")
    yyyymmdd = day.strftime("%Y%m%d")

    candidates = [
        f"{GOES_LEGACY_BASE}/{sat}/magnetometer-l2-avg1m/{yyyy}/{mm}/dn_magn-l2-avg1m_g{sat_num}_d{yyyymmdd}_v2-0-0.nc",
        f"{GOES_LEGACY_BASE}/{sat}/magnetometer-l2-avg1m/{yyyy}/{mm}/dn_magn-l2-avg1m_g{sat_num}_d{yyyymmdd}.nc",
        f"{GOES_LEGACY_BASE}/{sat}/magnetometer-l2/{yyyy}/{mm}/dn_magn-l2-avg1m_g{sat_num}_d{yyyymmdd}_v2-0-0.nc",
    ]
    return candidates


def ensure_daily_file(sat: str, day: datetime) -> Path:
    """
    Download one day of data into cache and return the local path.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = day.strftime("%Y%m%d")
    cached = list(CACHE_DIR.glob(f"{sat}_{tag}*.nc"))
    if cached:
        return cached[0]

    if is_goesr_satellite(sat):
        url = goesr_daily_url(sat, day)
        out = CACHE_DIR / f"{sat}_{tag}.nc"
        if download_file(url, out):
            return out
        raise FileNotFoundError(f"Could not download GOES-R file: {url}")

    for url in legacy_daily_url_candidates(sat, day):
        out = CACHE_DIR / f"{sat}_{tag}.nc"
        if download_file(url, out):
            return out

    raise FileNotFoundError(
        f"Could not download legacy file for {sat} on {tag}. "
        f"Adjust legacy_daily_url_candidates()."
    )


# ----------------------------
# NETCDF EXTRACTION
# ----------------------------

def find_time_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        low = col.lower()
        if low in {"time", "time_tag", "time_utc", "datetime"}:
            return col
        if "time" in low:
            return col
    raise KeyError("No time column found in dataset.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to map NOAA variable names into a simple standard set.
    """
    rename_map = {}

    for col in df.columns:
        low = col.lower()

        if low in {"hp", "h_p"}:
            rename_map[col] = "Hp"
        elif low in {"he", "h_e"}:
            rename_map[col] = "He"
        elif low in {"hn", "h_n"}:
            rename_map[col] = "Hn"
        elif low in {"bt", "b_total", "total_field"}:
            rename_map[col] = "Bt"

    df = df.rename(columns=rename_map)
    return df


def open_daily_as_dataframe(nc_path: Path) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    df = ds.to_dataframe().reset_index()
    df = normalize_columns(df)

    time_col = find_time_column(df)
    df["time_utc"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"])

    keep = ["time_utc"] + [c for c in ["Hp", "He", "Hn", "Bt"] if c in df.columns]
    df = df[keep].copy()
    df = df.sort_values("time_utc").drop_duplicates(subset=["time_utc"])
    return df


# ----------------------------
# WINDOW EXTRACTION
# ----------------------------

def daterange(start_day: datetime, end_day: datetime) -> Iterable[datetime]:
    day = start_day
    while day <= end_day:
        yield day
        day += timedelta(days=1)


def extract_window(center_time: datetime, window_hours: int) -> tuple[datetime, datetime]:
    start = center_time - timedelta(hours=window_hours)
    end = center_time + timedelta(hours=window_hours)
    return start, end


def build_event_csv(
    event_id: str,
    center_time: datetime,
    window_hours: int,
    channels: list[str],
) -> Path:
    sat = east_satellite_for_time(center_time)
    start, end = extract_window(center_time, window_hours)

    start_day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = end.replace(hour=0, minute=0, second=0, microsecond=0)

    pieces: list[pd.DataFrame] = []
    for day in daterange(start_day, end_day):
        local_nc = ensure_daily_file(sat, day)
        pieces.append(open_daily_as_dataframe(local_nc))

    df = pd.concat(pieces, ignore_index=True)
    df = df[(df["time_utc"] >= start) & (df["time_utc"] <= end)].copy()

    keep_cols = ["time_utc"] + [c for c in channels if c in df.columns]
    df = df[keep_cols].copy()

    df["event_id"] = event_id
    df["center_time_utc"] = center_time.isoformat()
    df["window_hours_each_side"] = window_hours
    df["source_satellite"] = sat

    ordered = [
        "event_id",
        "center_time_utc",
        "window_hours_each_side",
        "source_satellite",
        "time_utc",
    ] + [c for c in channels if c in df.columns]
    df = df[ordered]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / f"{event_id}_{sat}_{center_time.strftime('%Y%m%dT%H%MZ')}.csv"
    df.to_csv(outpath, index=False)
    return outpath


# ----------------------------
# MAIN
# ----------------------------

def main() -> int:
    events = load_events(INPUT_FILE)

    produced = []
    for _, row in events.iterrows():
        event_id = row["event_id"]
        center_time = row["center_time_utc"]
        window_hours = int(row.get("window_hours", DEFAULT_WINDOW_HOURS))

        raw_channels = row.get("channels", "|".join(DEFAULT_CHANNELS))
        channels = [c.strip() for c in str(raw_channels).split("|") if c.strip()]

        try:
            out = build_event_csv(
                event_id=event_id,
                center_time=center_time,
                window_hours=window_hours,
                channels=channels,
            )
            print(f"[OK] {event_id} -> {out}")
            produced.append(str(out))
        except Exception as e:
            print(f"[FAIL] {event_id}: {e}", file=sys.stderr)

    print(f"\nDone. Created {len(produced)} CSV files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
