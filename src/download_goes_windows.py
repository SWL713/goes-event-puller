from __future__ import annotations

import calendar
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests
import xarray as xr


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
SESSION.headers.update({"User-Agent": "goes-event-puller/4.0"})


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


def download_file(url: str, outpath: Path) -> bool:
    print(f"Downloading: {url}")
    response = SESSION.get(url, timeout=TIMEOUT)
    if response.status_code == 200:
        outpath.write_bytes(response.content)
        return True
    print(f"  -> HTTP {response.status_code}", file=sys.stderr)
    return False


def fetch_text(url: str) -> str | None:
    print(f"Listing: {url}")
    response = SESSION.get(url, timeout=TIMEOUT)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text


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


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_time_column(df: pd.DataFrame) -> str:
    candidates = []
    for col in df.columns:
        n = normalize_name(col)
        if n in {"time", "timetag", "timeutc", "datetime"}:
            candidates.append(col)
        elif "time" in n:
            candidates.append(col)

    if candidates:
        return candidates[0]

    raise KeyError(f"No time column found. Columns: {list(df.columns)}")


def classify_magnetic_column(col: str) -> str | None:
    n = normalize_name(col)

    if n in {"hp", "hpmag", "hpavg", "hpavg1m"}:
        return "Hp"
    if n in {"he", "hemag", "heavg", "heavg1m"}:
        return "He"
    if n in {"hn", "hnmag", "hnavg", "hnavg1m"}:
        return "Hn"
    if n in {"bt", "ht", "btotal", "totalfield", "btavg", "htavg"}:
        return "Bt"

    return None


def open_netcdf_as_dataframe(nc_path: Path) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    df = ds.to_dataframe().reset_index()

    rename_map = {}
    for col in df.columns:
        mapped = classify_magnetic_column(col)
        if mapped is not None:
            rename_map[col] = mapped

    df = df.rename(columns=rename_map)

    time_col = find_time_column(df)
    df["time_utc"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"]).copy()

    for col in ["Hp", "He", "Hn", "Bt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = ["time_utc"] + [c for c in ["Hp", "He", "Hn", "Bt"] if c in df.columns]
    df = df[keep_cols].copy()
    df = df.sort_values("time_utc").drop_duplicates(subset=["time_utc"])

    return df


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

    return df, sat


def process_event(row: pd.Series) -> dict:
    event_id = str(row["event_id"])
    center_time = row["center_time_utc"]
    window_hours = int(row["window_hours"])
    channels = [c.strip() for c in str(row["channels"]).split("|") if c.strip()]

    event_dir = OUTPUT_DIR / event_id
    event_dir.mkdir(parents=True, exist_ok=True)

    df, sat = load_event_window(event_id, center_time, window_hours, channels)

    raw_out = df.copy()
    raw_out["event_id"] = event_id
    raw_out["center_time_utc"] = center_time.isoformat()
    raw_out["source_satellite"] = sat
    raw_out.to_csv(event_dir / f"{event_id}_goes_window.csv", index=False)

    present_cols = [c for c in ["Hp", "He", "Hn", "Bt"] if c in df.columns]

    summary = {
        "event_id": event_id,
        "center_time_utc": center_time.isoformat(),
        "window_hours_each_side": window_hours,
        "source_satellite": sat,
        "n_rows": int(len(df)),
        "present_columns": present_cols,
        "time_start_utc": df["time_utc"].min().isoformat() if len(df) else None,
        "time_end_utc": df["time_utc"].max().isoformat() if len(df) else None,
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

    return 0 if created > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
