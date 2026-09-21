#!/usr/bin/env python3
"""Download NASA POWER station benchmarks and compare Tianji/ERA5 point series."""

from __future__ import annotations

import argparse
import calendar
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


API_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMETERS = ["ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "WD10M"]


def month_bounds(month: str) -> tuple[str, str]:
    year, mon = [int(part) for part in month.split("-")]
    last = calendar.monthrange(year, mon)[1]
    return f"{year:04d}{mon:02d}01", f"{year:04d}{mon:02d}{last:02d}"


def fetch_power(lat: float, lon: float, month: str, timeout: int) -> dict[str, object]:
    start, end = month_bounds(month)
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "RE",
        "longitude": f"{lon:.5f}",
        "latitude": f"{lat:.5f}",
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "UTC",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params, safe=",")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    payload["_request_url"] = url
    return payload


def normalize_power(payload: dict[str, object], station_id: str, month: str) -> pd.DataFrame:
    params = payload.get("properties", {}).get("parameter", {})  # type: ignore[union-attr]
    if not isinstance(params, dict) or not params:
        return pd.DataFrame()
    rows: dict[str, pd.Series] = {}
    for name, values in params.items():
        if not isinstance(values, dict):
            continue
        series = pd.Series(values, name=name)
        rows[name] = pd.to_numeric(series, errors="coerce")
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows.values(), axis=1)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H", errors="coerce")
    df.index.name = "time_utc"
    df = df.reset_index().dropna(subset=["time_utc"])
    df.insert(0, "station_id", str(station_id))
    df.insert(1, "month", month)
    df = df.rename(
        columns={
            "ALLSKY_SFC_SW_DWN": "ghi_wm2",
            "T2M": "t2m_c",
            "WS10M": "wind_speed_ms",
            "WD10M": "wind_dir_deg",
        }
    )
    # NASA POWER reports ALLSKY_SFC_SW_DWN as Wh/m^2 per hour. For hourly data,
    # this is numerically equal to the mean W/m^2 over that hour.
    for col in ["ghi_wm2", "t2m_c", "wind_speed_ms", "wind_dir_deg"]:
        if col in df:
            df.loc[df[col] <= -900, col] = np.nan
    df["source"] = "NASA_POWER_hourly_point"
    return df


def choose_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def dataset_specs(month: str) -> list[dict[str, object]]:
    return [
        {
            "dataset": "Tianji_2.5km",
            "root": choose_existing(
                [
                    Path(f"data/processed/tianji_monthly/{month}"),
                    Path(f"data/processed/tianji_monthly/{month}_partial"),
                ]
            ),
        },
        {
            "dataset": "ERA5_regridded_2.5km",
            "root": choose_existing(
                [
                    Path(f"data/processed/era5_on_tianji_monthly/{month}"),
                    Path(f"data/processed/era5_on_tianji_monthly/{month}_partial"),
                ]
            ),
        },
        {
            "dataset": "ERA5_native_0.25deg",
            "root": choose_existing(
                [
                    Path(f"data/processed/era5_monthly/{month}"),
                    Path(f"data/processed/era5_monthly/{month}_partial"),
                ]
            ),
        },
    ]


def sample_nc(root: Path, var: str, lon: float, lat: float, method: str) -> pd.Series | None:
    path = root / f"{var}.nc"
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        da = ds[var] if var in ds.data_vars else next(iter(ds.data_vars.values()))
        if method == "nearest":
            sample = da.sel(grid_xt=lon, grid_yt=lat, method="nearest")
        else:
            sample = da.interp(grid_xt=lon, grid_yt=lat, method="linear")
        series = sample.to_series()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = pd.to_numeric(series, errors="coerce")
    if var == "t2mz" and series.median(skipna=True) > 150:
        series = series - 273.15
    return series


def metrics(obs: pd.Series, sim: pd.Series) -> dict[str, float]:
    aligned = pd.concat([obs.rename("obs"), sim.rename("sim")], axis=1).dropna()
    if aligned.empty:
        return {"n": 0, "mbe": math.nan, "mae": math.nan, "rmse": math.nan, "corr": math.nan, "r2": math.nan}
    diff = aligned["sim"] - aligned["obs"]
    corr = aligned["obs"].corr(aligned["sim"]) if len(aligned) > 1 else math.nan
    return {
        "n": int(len(aligned)),
        "mbe": float(diff.mean()),
        "mae": float(diff.abs().mean()),
        "rmse": float(np.sqrt((diff**2).mean())),
        "corr": float(corr) if pd.notna(corr) else math.nan,
        "r2": float(corr**2) if pd.notna(corr) else math.nan,
    }


def ramp_metrics(obs: pd.Series, sim: pd.Series, threshold: float, tolerance_h: float) -> dict[str, float]:
    aligned = pd.concat([obs.rename("obs"), sim.rename("sim")], axis=1).dropna()
    if aligned.empty:
        return {"obs_ramps": 0, "sim_ramps": 0, "pod": math.nan, "far": math.nan, "csi": math.nan}
    obs_events = aligned["obs"].diff().abs()
    sim_events = aligned["sim"].diff().abs()
    obs_times = obs_events[obs_events >= threshold].dropna().index
    sim_times = sim_events[sim_events >= threshold].dropna().index
    used: set[pd.Timestamp] = set()
    hits = 0
    tol = pd.Timedelta(hours=tolerance_h)
    for t in obs_times:
        candidates = [s for s in sim_times if s not in used and abs(s - t) <= tol]
        if candidates:
            nearest = min(candidates, key=lambda s: abs(s - t))
            used.add(nearest)
            hits += 1
    obs_n = len(obs_times)
    sim_n = len(sim_times)
    false_alarms = sim_n - hits
    misses = obs_n - hits
    return {
        "obs_ramps": int(obs_n),
        "sim_ramps": int(sim_n),
        "pod": float(hits / obs_n) if obs_n else math.nan,
        "far": float(false_alarms / sim_n) if sim_n else math.nan,
        "csi": float(hits / (hits + misses + false_alarms)) if (hits + misses + false_alarms) else math.nan,
    }


def write_report(metrics_df: pd.DataFrame, ramp_df: pd.DataFrame, report: Path) -> None:
    valid = metrics_df[metrics_df["n"].fillna(0).gt(0)] if not metrics_df.empty else pd.DataFrame()
    lines = [
        "# NASA POWER Independent Product Benchmark",
        "",
        "benchmark_status = metrics_generated" if not valid.empty else "benchmark_status = no_overlap",
        "claim_boundary = independent_product_benchmark_not_ground_truth_validation",
        "",
        "NASA POWER hourly `ALLSKY_SFC_SW_DWN` is reported as Wh/m^2 per hour; it is used here as hourly mean W/m^2 for comparison with model GHI.",
        "",
        "## Metric Summary",
        valid.to_markdown(index=False) if not valid.empty else "No matched samples were available.",
        "",
        "## GHI Ramp Skill Against NASA POWER",
        ramp_df.to_markdown(index=False) if not ramp_df.empty else "No ramp benchmark metrics generated.",
        "",
        "## Interpretation Boundary",
        "",
        "This is an independent open-product benchmark. It is useful for plausibility and reproducibility, but it does not replace measured CMA radiation, pyranometer, PVOD, or plant SCADA validation.",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-metadata", default="data/external_validation/station_metadata_template.csv")
    parser.add_argument("--months", nargs="+", default=["2025-04", "2025-07", "2025-10"])
    parser.add_argument("--output-dir", default="data/external_validation/nasa_power")
    parser.add_argument("--analysis-dir", default="data/analysis/nasa_power_benchmark")
    parser.add_argument("--report", default="reports/nasa_power_independent_benchmark.md")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--method", choices=["nearest", "linear"], default="linear")
    parser.add_argument("--ghi-ramp-threshold", type=float, default=200.0)
    parser.add_argument("--timing-tolerance-h", type=float, default=1.0)
    args = parser.parse_args()

    stations = pd.read_csv(args.station_metadata)
    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    obs_frames = []
    for st in stations.itertuples():
        station_id = str(st.station_id)
        for month in args.months:
            station_dir = output_dir / month / station_id
            station_dir.mkdir(parents=True, exist_ok=True)
            raw_path = station_dir / "nasa_power_raw.json"
            csv_path = station_dir / "nasa_power_hourly.csv"
            if raw_path.exists() and csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=["time_utc"])
            else:
                payload = fetch_power(float(st.lat), float(st.lon), month, args.timeout)
                raw_payload = {k: v for k, v in payload.items() if k != "_request_url"}
                raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                (station_dir / "request_url.txt").write_text(str(payload["_request_url"]), encoding="utf-8")
                df = normalize_power(payload, station_id, month)
                df.to_csv(csv_path, index=False, encoding="utf-8")
            obs_frames.append(df)

    obs = pd.concat(obs_frames, ignore_index=True) if obs_frames else pd.DataFrame()
    obs.to_csv(analysis_dir / "nasa_power_station_hourly_all.csv", index=False, encoding="utf-8")

    metric_rows = []
    ramp_rows = []
    for st in stations.itertuples():
        station_id = str(st.station_id)
        station_name = getattr(st, "station_name", station_id)
        for month in args.months:
            station_obs = obs[(obs["station_id"].astype(str) == station_id) & (obs["month"] == month)].copy()
            if station_obs.empty:
                continue
            station_obs["time_utc"] = pd.to_datetime(station_obs["time_utc"]).dt.tz_localize(None)
            station_obs = station_obs.set_index("time_utc")
            for spec in dataset_specs(month):
                root = spec["root"]
                if root is None:
                    continue
                for model_var, obs_var, variable in [
                    ("DSWRFsfc", "ghi_wm2", "ghi_wm2"),
                    ("t2mz", "t2m_c", "t2m_c"),
                ]:
                    sim = sample_nc(Path(root), model_var, float(st.lon), float(st.lat), args.method)
                    if sim is None:
                        continue
                    row = {
                        "station_id": station_id,
                        "station_name": station_name,
                        "month": month,
                        "dataset": spec["dataset"],
                        "method": args.method,
                        "variable": variable,
                    }
                    row.update(metrics(station_obs[obs_var], sim))
                    metric_rows.append(row)
                    if variable == "ghi_wm2" and row["n"] > 0:
                        rrow = {k: row[k] for k in ["station_id", "station_name", "month", "dataset", "method", "variable"]}
                        rrow.update(ramp_metrics(station_obs[obs_var], sim, args.ghi_ramp_threshold, args.timing_tolerance_h))
                        rrow["ramp_threshold_wm2"] = args.ghi_ramp_threshold
                        rrow["timing_tolerance_h"] = args.timing_tolerance_h
                        ramp_rows.append(rrow)

    metrics_df = pd.DataFrame(metric_rows)
    ramp_df = pd.DataFrame(ramp_rows)
    metrics_df.to_csv(analysis_dir / "nasa_power_model_metrics.csv", index=False, encoding="utf-8")
    ramp_df.to_csv(analysis_dir / "nasa_power_ghi_ramp_metrics.csv", index=False, encoding="utf-8")
    write_report(metrics_df, ramp_df, Path(args.report))
    print(f"[OK] wrote {analysis_dir / 'nasa_power_station_hourly_all.csv'} rows={len(obs)}")
    print(f"[OK] wrote {analysis_dir / 'nasa_power_model_metrics.csv'} rows={len(metrics_df)}")
    print(f"[OK] wrote {analysis_dir / 'nasa_power_ghi_ramp_metrics.csv'} rows={len(ramp_df)}")
    print(f"[OK] wrote {args.report}")


if __name__ == "__main__":
    main()
