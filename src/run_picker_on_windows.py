from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from picker_v5_baseline import detect_candidates


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "output"


def make_event_plot(trace_df: pd.DataFrame, detections_df: pd.DataFrame, title: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(trace_df["time_utc"], trace_df["Hp"], label="Hp raw", linewidth=0.8)
    ax.plot(trace_df["time_utc"], trace_df["hp_smooth"], label="Hp smooth", linewidth=1.5)

    if len(detections_df) > 0:
        ymax = float(trace_df["hp_smooth"].max())
        ymin = float(trace_df["hp_smooth"].min())
        yspan = ymax - ymin if ymax != ymin else 1.0
        label_y = ymax - 0.03 * yspan

        for _, row in detections_df.iterrows():
            g_time = row["g_time"]
            ax.axvline(g_time, linestyle="--", linewidth=1.0)
            ax.text(
                g_time,
                label_y,
                pd.Timestamp(g_time).strftime("%m-%d %H:%M"),
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xlabel("UTC")
    ax.set_ylabel("Hp")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def write_review_template(outpath: Path, event_id: str, detections_df: pd.DataFrame) -> None:
    rows = []

    for _, row in detections_df.iterrows():
        rows.append(
            {
                "event_id": event_id,
                "time_utc": row["g_time"].isoformat(),
                "review_status": "",
                "confidence": "",
                "notes": "",
            }
        )

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


def process_event_dir(event_dir: Path) -> dict | None:
    event_id = event_dir.name
    window_file = event_dir / f"{event_id}_goes_window.csv"

    if not window_file.exists():
        return None

    df = pd.read_csv(window_file)
    if "time_utc" not in df.columns:
        raise ValueError(f"{window_file} is missing time_utc")

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"]).copy()

    trace_df, detections_df = detect_candidates(df)

    detections_df.to_csv(event_dir / f"{event_id}_g_candidates.csv", index=False)
    trace_df.to_csv(event_dir / f"{event_id}_trace_used.csv", index=False)

    center_time = None
    if "center_time_utc" in df.columns and df["center_time_utc"].notna().any():
        center_time = str(df["center_time_utc"].dropna().iloc[0])

    title = f"{event_id} | GOES-East historical review"
    if center_time:
        title += f" | center={center_time.replace('+00:00', ' UTC')}"

    make_event_plot(trace_df, detections_df, title, event_dir / f"{event_id}_plot.png")
    write_review_template(event_dir / f"{event_id}_review_template.csv", event_id, detections_df)

    summary = {
        "event_id": event_id,
        "n_candidates": int(len(detections_df)),
        "candidate_times_utc": [pd.Timestamp(t).isoformat() for t in detections_df["g_time"]] if len(detections_df) > 0 else [],
    }

    with open(event_dir / f"{event_id}_picker_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> int:
    event_dirs = sorted([p for p in OUTPUT_DIR.iterdir() if p.is_dir()])

    summaries = []
    failures = []

    for event_dir in event_dirs:
        try:
            result = process_event_dir(event_dir)
            if result is not None:
                summaries.append(result)
        except Exception as exc:
            failures.append({"event_id": event_dir.name, "error": str(exc)})

    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "picker_batch_summary.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(OUTPUT_DIR / "picker_batch_failures.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
