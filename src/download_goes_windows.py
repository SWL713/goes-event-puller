from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests
import xarray as xr


# ============================================================
# PATHS / CONFIG
# ============================================================

# Because this file lives in src/, parent.parent points to repo root
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "events.csv"
OUTPUT_DIR = BASE_DIR / "data" / "output"
CACHE_DIR = BASE_DIR / "data" / "cache"

DEFAULT_WINDOW_HOURS = 24
DEFAULT_CHANNELS = ["Hp", "Bt"]
TIMEOUT = 60

# Base path for modern GOES-R/U NetCDF files
# Example pattern:
# https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/goes16/magnetometer-l2-avg1m/2024/05/dn_magn-l2-avg1m_g16_d20240510_v2-0-0.nc
GOESR_BASE = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access"

# Placeholder base for legacy satellites.
# Older GOES archive layout can vary. This script is structured so you only need
# to adjust legacy_daily_url_candidates() if legacy downloads fail.
LEGACY_BASE = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access"

# Make sure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Shared session
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "goes-event-puller/1.0"})


# ============================================================
# SATELLITE MAPPING
# ============================================================

def east_satellite_for_time(ts: datetime) -> str:
    """
    Returns the GOES-East spacecraft assumed for a given UTC timestamp.

    This is intentionally centralized so you can edit it later if needed.

    Current mapping used here:
    - before 2003-04-01 : goes08
    - before 2010-04-14 : goes12
    - before 2017-12-18 : goes13
    - before 2025-04-07 : goes16
    - otherwise         : goes19
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


# ============================================================
# INPUT PARSING
# ============================================================

def parse_user_timestamp(value: str) -> datetime:
    """
    Accepts dates from events.csv such as:
    5/15/2005 6:35
    10/10/2024 22:32
    8/24/2005 9:11

    Assumes UTC.
    """
    text = str(value).strip()

    formats = [
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%y %H:%M",
        "%m/%d/%y %H:%M:%S",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

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
# DOWNLOAD HELPERS
# ============================================================

def download_file(url: str, outpath: Path) -> bool:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {url}")
    response = SESSION.get(url, timeout=TIMEOUT)

    if response.status_code == 200:
        outpath.write_bytes(response.content)
        return True

    print(f"  -> HTTP {response.status_code}", file=sys.stderr)
    return False


# ============================================================
# URL BUILDERS
# ============================================================

def goesr_daily_url(sat: str, day: datetime) -> str:
    """
    Build the standard GOES-R/U avg1m magnetometer daily file URL.
    """
    sat_num = sat.replace("goes", "")
    yyyy = day.strftime("%Y")
    mm = day.strftime("%m")
    yyyymmdd = day.strftime("%Y%m%d")

    filename = f"dn_magn-l2-avg1m_g{sat_num}_d{yyyymmdd}_v2-0-0.nc"
    return f"{GOESR_BASE}/{sat}/magnetometer-l2-avg1m/{yyyy}/{mm}/{filename}"


def legacy_daily_url_candidates(sat: str, day: datetime) -> List[str]:
    """
    Candidate legacy paths for older GOES spacecraft.

    You may need to tweak these if older downloads fail.
    The rest of the script does not need to change.
    """
    sat_num = sat.replace("goes", "")
    yyyy = day.strftime("%Y")
    mm = day.strftime("%m")
    yyyymmdd = day.strftime("%Y%m%d")

    return [
        f"{LEGACY_BASE}/{sat}/magnetometer-l2-avg1m/{yyyy}/{mm}/dn_magn-l2-avg1m_g{sat_num}_d{yyyymmdd}_v2-0-0.nc",
        f"{LEGACY_BASE}/{sat}/magnetometer-l2-avg1m/{yyyy}/{mm}/dn_magn-l2-avg1m_g{sat_num}_d{yyyymmdd}.nc",
        f"{LEGACY_BASE}/{sat}/magnetometer-l2/{yyyy}/{mm}/dn_magn-l2-avg1m_g{sat_num}_d{yyyymmdd}_v2-0-0.nc",
        f"{LEGACY_BASE}/{sat}/magnetometer-l2/{yyyy}/{mm}/dn_magn-l2-avg1m_g{sat_num}_d{yyyymmdd}.nc",
    ]


def ensure_daily_file(sat: str, day: datetime) -> Path:
    """
    Ensures one daily NetCDF file exists in cache and returns its local path.
    """
    tag = day.strftime("%Y%m%d")
    cached_path = CACHE_DIR / f"{sat}_{tag}.nc"

    if cached_path.exists():
        print(f"Using cache: {cached_path.name}")
        return cached_path

    if is_goesr_satellite(sat):
        url = goesr_daily_url(sat, day)
        if download_file(url, cached_path):
            return cached_path
        raise FileNotFoundError(f"Could not download GOES-R/U file: {url}")

    for url in legacy_daily_url_candidates(sat, day):
        if download_file(url, cached_path):
            return cached_path

    raise FileNotFoundError(
        f"Could not download legacy file for {sat} on {tag}. "
        f"Update legacy_daily_url_candidates() with the correct path pattern."
    )


# ============================================================
# NETCDF HANDLING
# ============================================================

def find_time_column(df: pd.DataFrame) -> str:
    """
    Try to locate the time column after xarray converts the dataset.
    """
    preferred = ["time", "time_tag", "time_utc", "datetime"]

    lower_map = {col.lower(): col for col in df.columns}
    for name in preferred:
        if name in lower_map:
            return lower_map[name]

    for col in df.columns:
        if "time" in col.lower():
            return col

    raise KeyError(f"No time column found. Columns were: {list(df.columns)}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map possible NOAA field names to a simple standard set.
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

    return df.rename(columns=rename_map)


def open_daily_as_dataframe(nc_path: Path) -> pd.DataFrame:
    """
    Open one NetCDF daily file and return a normalized dataframe.
    """
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
    day = start_day
    while day <= end_day:
        yield day
        day += timedelta(days=1)


def extract_window(center_time: datetime, window_hours: int) -> tuple[datetime, datetime]:
    """
    window_hours means HOURS ON EACH SIDE.
    So 24 means center_time ± 24 hours.
    """
    start = center_time - timedelta(hours=window_hours)
    end = center_time + timedelta(hours=window_hours)
    return start, end


def build_event_csv(
    event_id: str,
    center_time: datetime,
    window_hours: int,
    channels: List[str],
) -> Path:
    sat = east_satellite_for_time(center_time)
    start, end = extract_window(center_time, window_hours)

    start_day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = end.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"\nProcessing {event_id}")
    print(f"  center_time_utc: {center_time.isoformat()}")
    print(f"  source_satellite: {sat}")
    print(f"  window: {start.isoformat()}  ->  {end.isoformat()}")

    pieces = []
    for day in daterange(start_day, end_day):
        local_nc = ensure_daily_file(sat, day)
        day_df = open_daily_as_dataframe(local_nc)
        pieces.append(day_df)

    df = pd.concat(pieces, ignore_index=True)
    df = df[(df["time_utc"] >= start) & (df["time_utc"] <= end)].copy()

    available_channels = [c for c in channels if c in df.columns]
    missing_channels = [c for c in channels if c not in df.columns]
    if missing_channels:
        print(f"  warning: missing channels in file: {missing_channels}", file=sys.stderr)

    keep_cols = ["time_utc"] + available_channels
    df = df[keep_cols].copy()

    df["event_id"] = event_id
    df["center_time_utc"] = center_time.isoformat()
    df["window_hours_each_side"] = window_hours
    df["source_satellite"] = sat

    ordered_cols = [
        "event_id",
        "center_time_utc",
        "window_hours_each_side",
        "source_satellite",
        "time_utc",
    ] + available_channels
    df = df[ordered_cols]

    outname = f"{event_id}_{sat}_{center_time.strftime('%Y%m%dT%H%MZ')}.csv"
    outpath = OUTPUT_DIR / outname
    df.to_csv(outpath, index=False)

    print(f"  wrote: {outpath}")
    return outpath


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print(f"Reading events from: {INPUT_FILE}")
    events = load_events(INPUT_FILE)

    created = 0
    failed = 0

    for _, row in events.iterrows():
        event_id = str(row["event_id"])
        center_time = row["center_time_utc"]
        window_hours = int(row["window_hours"])
        channels = [c.strip() for c in str(row["channels"]).split("|") if c.strip()]

        try:
            build_event_csv(
                event_id=event_id,
                center_time=center_time,
                window_hours=window_hours,
                channels=channels,
            )
            created += 1
        except Exception as exc:
            failed += 1
            print(f"\n[FAIL] {event_id}: {exc}", file=sys.stderr)

    print("\nFinished.")
    print(f"  created: {created}")
    print(f"  failed:  {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
