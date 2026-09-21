#!/usr/bin/env python3
"""Detect gridded hourly PV ramp events."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def build_valid_mask(
    pv: xr.DataArray,
    ghi: xr.DataArray | None,
    min_ghi: float,
    min_pv: float,
) -> tuple[xr.DataArray, pd.DataFrame]:
    delta = pv.diff("time").rename("delta_p")
    p_prev = pv.isel(time=slice(None, -1))
    p_now = pv.isel(time=slice(1, None))

    valid = xr.ones_like(delta, dtype=bool)
    if min_pv > 0:
        valid = valid & (np.maximum(p_prev, p_now) >= min_pv)

    if ghi is not None and min_ghi > 0:
        ghi_prev = ghi.isel(time=slice(None, -1))
        ghi_now = ghi.isel(time=slice(1, None))
        valid = valid & (np.maximum(ghi_prev, ghi_now) >= min_ghi)

    active_ratio = valid.mean(dim=("grid_yt", "grid_xt"))
    diagnostics = pd.DataFrame(
        {
            "time": pd.to_datetime(active_ratio["time"].values),
            "valid_daylight_ratio": active_ratio.values,
        }
    )

    return valid, diagnostics


def detect_for_threshold(
    pv: xr.DataArray,
    threshold: float,
    valid_mask: xr.DataArray,
    max_simultaneity: float | None,
) -> pd.DataFrame:
    delta = pv.diff("time").rename("delta_p")
    p_now = pv.isel(time=slice(1, None))
    up_mask = delta >= threshold
    down_mask = delta <= -threshold
    event_mask = (up_mask | down_mask) & valid_mask
    if max_simultaneity is not None:
        event_ratio = event_mask.mean(dim=("grid_yt", "grid_xt"))
        event_mask = event_mask & (event_ratio <= max_simultaneity)

    event_flat = event_mask.stack(event_id=("time", "grid_yt", "grid_xt"))
    event_indices = np.flatnonzero(event_flat.values)
    stacked = xr.Dataset({"p_pu": p_now, "delta_p": delta}).stack(event_id=("time", "grid_yt", "grid_xt")).isel(event_id=event_indices)
    if stacked.sizes.get("event_id", 0) == 0:
        return pd.DataFrame(columns=["time", "lon", "lat", "grid_x", "grid_y", "p_pu", "delta_p", "ramp_type", "threshold"])

    df = stacked[["p_pu", "delta_p"]].to_dataframe().reset_index()
    df = df.rename(columns={"grid_xt": "lon", "grid_yt": "lat"})
    lon_values = pv["grid_xt"].values
    lat_values = pv["grid_yt"].values
    lon_to_ix = {round(float(v), 10): i for i, v in enumerate(lon_values)}
    lat_to_iy = {round(float(v), 10): i for i, v in enumerate(lat_values)}
    df["grid_x"] = df["lon"].map(lambda x: lon_to_ix[round(float(x), 10)])
    df["grid_y"] = df["lat"].map(lambda y: lat_to_iy[round(float(y), 10)])
    df["ramp_type"] = np.where(df["delta_p"] >= threshold, "up", "down")
    df["threshold"] = threshold
    return df[["time", "lon", "lat", "grid_x", "grid_y", "p_pu", "delta_p", "ramp_type", "threshold"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weather-root", help="Processed Tianji weather root containing DSWRFsfc.nc")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.3, 0.5])
    parser.add_argument("--min-ghi", type=float, default=100.0, help="Minimum max(GHI_t-1, GHI_t) for valid daytime ramps")
    parser.add_argument("--min-pv", type=float, default=0.1, help="Minimum max(PV_t-1, PV_t) for valid daytime ramps")
    parser.add_argument(
        "--max-simultaneity",
        type=float,
        default=0.8,
        help="Drop hours where threshold-specific ramp cells exceed this domain fraction; set >=1 to disable",
    )
    parser.add_argument("--diagnostics-output", help="Optional CSV with hourly daylight diagnostics")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    max_simultaneity = args.max_simultaneity if args.max_simultaneity < 1 else None

    with xr.open_dataset(args.input) as ds:
        pv = ds["pv_pu"].load()

    ghi = None
    if args.weather_root:
        ghi_path = Path(args.weather_root) / "DSWRFsfc.nc"
        with xr.open_dataset(ghi_path) as ghi_ds:
            ghi_name = "DSWRFsfc" if "DSWRFsfc" in ghi_ds.data_vars else next(iter(ghi_ds.data_vars))
            ghi = ghi_ds[ghi_name].load()

    valid_mask, diagnostics = build_valid_mask(
        pv=pv,
        ghi=ghi,
        min_ghi=args.min_ghi,
        min_pv=args.min_pv,
    )
    frames = [detect_for_threshold(pv, threshold, valid_mask, max_simultaneity) for threshold in args.thresholds]

    result = pd.concat(frames, ignore_index=True)
    if not result.empty:
        result["time"] = pd.to_datetime(result["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    result.to_csv(output, index=False, encoding="utf-8")

    if args.diagnostics_output:
        diagnostics_output = Path(args.diagnostics_output)
        diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        if not diagnostics.empty:
            diagnostics["time"] = pd.to_datetime(diagnostics["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        diagnostics.to_csv(diagnostics_output, index=False, encoding="utf-8")
        print(f"[OK] wrote {diagnostics_output}")

    summary = result.groupby(["threshold", "ramp_type"]).size().reset_index(name="events") if not result.empty else pd.DataFrame()
    print(f"[OK] wrote {output}")
    print(f"filters: min_ghi={args.min_ghi}, min_pv={args.min_pv}, max_simultaneity={args.max_simultaneity}")
    print(summary.to_string(index=False) if not summary.empty else "no events")


if __name__ == "__main__":
    main()
