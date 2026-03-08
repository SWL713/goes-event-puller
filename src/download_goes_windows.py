from __future__ import annotations

import calendar
import re
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

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "events.csv"
OUTPUT_DIR = BASE_DIR / "data" / "output"
CACHE_DIR = BASE_DIR / "data" / "cache"

DEFAULT_WINDOW_HOURS = 24
DEFAULT_CHANNELS = ["Hp", "Bt"]
TIMEOUT = 60

# Modern GOES-R / GOES-U archive
GOESR_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"

# Legacy GOES 8-15 archive
LEGACY_BASE = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "goes-event-puller/1.1"})


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


def fetch_text(url: str) -> str:
    print(f"Listing: {url}")
    r = SESSION.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


# ============================================================
# MODERN GOES-R / GOES-U
# ============================================================

def goesr_month_dir(sat: str, day: datetime) -> str:
    yyyy = day.strftime("%Y")
    mm = day.strftime("%m")
    return f"{GOESR_BASE}/{sat}/l2/data/magn-l2-avg1m/{yyyy}/{mm}/"


def find_goesr_filename(sat: str, day: datetime) -> str:
    """
    Discover the exact filename from the directory listing, so we do not
    hard-code a version suffix like v2-0-0 or v2-0-2.
    """
    sat_num = sat.replace("goes", "")
    yyyymmdd = day.strftime("%Y%m%d")
    month_url = goesr_month_dir(sat, day)
    html = fetch_text(month_url)

    pattern = rf"dn_magn-l2-avg1m_g{sat_num}_d{yyyymmdd}_v[\d\-]+\.nc"
    matches = re.findall(pattern, html)

    if not matches:
        raise FileNotFoundError(
            f"Could not find GOES-R/U file for {sat} on {yyyymmdd} in {month_url}"
        )

    # Usually exactly one match
    return sorted(set(matches))[-1]


def ensure_goesr_daily_file(sat: str, day: datetime) -> Path:
    tag = day.strftime("%Y%m%d")
    cached = CACHE_DIR / f"{sat}_{tag}.nc"
    if cached.exists():
        print(f"Using cache: {cached.name}")
        return cached

    filename = find_goesr_filename(sat, day)
    url = goesr_month_dir(sat, day) + filename
    if not download_file(url, cached):
        raise FileNotFoundError(f"Could not download modern file: {url}")
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


def ensure_legacy_month_file(sat: str, month_day: datetime) -> Path:
    tag = month_day.strftime("%Y%m")
    cached = CACHE_DIR / f"{sat}_{tag}.nc"
    if cached.exists():
        print(f"Using cache: {cached.name}")
        return cached

    filename = legacy_month_filename(sat, month_day)
    url = legacy_month_dir(sat, month_day) + filename
    if not download_file(url, cached):
        raise FileNotFoundError(f"Could not download legacy file: {url}")
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
            # Some files use Ht rather than Bt for total field.
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


def build_event_csv(
    event_id: str,
    center_time: datetime,
    window_hours: int,
    channels: List[str],
) -> Path:
    sat = east_satellite_for_time(center_time)
    start, end = extract_window(center_time, window_hours)

    print(f"\nProcessing {event_id}")
    print(f"  center_time_utc: {center_time.isoformat()}")
    print(f"  source_satellite: {sat}")
    print(f"  window: {start.isoformat()}  ->  {end.isoformat()}")

    pieces: List[pd.DataFrame] = []

    if is_goesr_satellite(sat):
        start_day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end_day = end.replace(hour=0, minute=0, second=0, microsecond=0)

        for day in daterange(start_day, end_day):
            local_nc = ensure_goesr_daily_file(sat, day)
            pieces.append(open_netcdf_as_dataframe(local_nc))
    else:
        for mon in month_range(start, end):
            local_nc = ensure_legacy_month_file(sat, mon)
            pieces.append(open_netcdf_as_dataframe(local_nc))

    df = pd.concat(pieces, ignore_index=True)
    df = df[(df["time_utc"] >= start) & (df["time_utc"] <= end)].copy()

    available = [c for c in channels if c in df.columns]
    missing = [c for c in channels if c not in df.columns]
    if missing:
        print(f"  warning: missing channels in file: {missing}", file=sys.stderr)

    df = df[["time_utc"] + available].copy()
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
    ] + available
    df = df[ordered]

    outpath = OUTPUT_DIR / f"{event_id}_{sat}_{center_time.strftime('%Y%m%dT%H%MZ')}.csv"
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
            build_event_csv(event_id, center_time, window_hours, channels)
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
