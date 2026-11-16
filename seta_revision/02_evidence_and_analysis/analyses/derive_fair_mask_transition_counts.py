#!/usr/bin/env python3
"""Record consecutive-transition counts for the all-product fair-hour mask."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "rerun_fair_mask_benchmark.py"
OUT = HERE / "fair_mask_common_hour_transition_counts.csv"
META = HERE / "fair_mask_common_hour_transition_counts_metadata.json"
MONTHS = ["2025-04", "2025-07"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("seta_hourly_rerun", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    rerun = load_module(SOURCE)
    stations = pd.read_csv(rerun.STATIONS, low_memory=False)
    stations["local_time"] = (
        pd.to_datetime(stations["datetime_hubei_local_fixed"], utc=True)
        .dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    )
    stations = stations[
        stations["common_period_flag"].eq(1)
        & stations["usable_absolute_event_pairing"].eq(1)
        & stations["solar_qc_group"].eq("primary_solar_qc")
    ]

    cache = {}
    rows = []
    for month in MONTHS:
        month_stations = stations[stations.local_time.dt.strftime("%Y-%m").eq(month)]
        for station_id, group in month_stations.groupby("station_id", sort=True):
            site = group.iloc[0]
            observed = pd.Series(
                group.sort_values("local_time").ghi_wm2.to_numpy(float),
                index=pd.DatetimeIndex(group.sort_values("local_time").local_time),
                name="obs",
            )
            model_series = {}
            for product in rerun.PRODUCTS:
                key = (product, month)
                if key not in cache:
                    cache[key] = rerun.read_field(product, month)
                model_series[product] = rerun.sampled_series(
                    product, cache[key], site
                ).rename(product)
            common = pd.concat([observed, *model_series.values()], axis=1).dropna()
            timestamps = pd.DatetimeIndex(common.index)
            transitions = int(
                timestamps.to_series().diff().eq(pd.Timedelta(hours=1)).sum()
            )
            rows.append({
                "station_id": int(station_id),
                "month": month,
                "valid_common_station_hours": len(common),
                "eligible_consecutive_within_month_transitions": transitions,
                "first_common_hour": timestamps.min().isoformat(),
                "last_common_hour": timestamps.max().isoformat(),
            })

    result = pd.DataFrame(rows).sort_values(["station_id", "month"])
    if len(result) != 14 or result.station_id.nunique() != 7:
        raise RuntimeError("Expected seven primary stations and 14 station-months")
    if int(result.valid_common_station_hours.sum()) != 9950:
        raise RuntimeError("Fair-mask hour total differs from the canonical benchmark")
    if int(result.eligible_consecutive_within_month_transitions.sum()) != 9925:
        raise RuntimeError("Unexpected consecutive-transition total")
    result.to_csv(OUT, index=False)
    metadata = {
        "analysis": "Consecutive within-month transition count on the all-seven-product fair mask",
        "months": MONTHS,
        "threshold_wm2h": 200,
        "station_time_shift_h": 0,
        "qc_group": "primary_solar_qc",
        "station_months": len(result),
        "stations": result.station_id.nunique(),
        "valid_common_station_hours": int(result.valid_common_station_hours.sum()),
        "eligible_consecutive_within_month_transitions": int(result.eligible_consecutive_within_month_transitions.sum()),
        "mask": "Station observation and all seven canonical product values valid at the timestamp; transition eligible only for exactly adjacent hourly timestamps within the same month.",
        "source_script": str(SOURCE),
        "source_script_sha256": sha256(SOURCE),
        "station_csv": str(rerun.STATIONS),
        "station_csv_sha256": sha256(rerun.STATIONS),
        "outputs": [OUT.name],
    }
    META.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(metadata["valid_common_station_hours"], metadata["eligible_consecutive_within_month_transitions"])


if __name__ == "__main__":
    main()
