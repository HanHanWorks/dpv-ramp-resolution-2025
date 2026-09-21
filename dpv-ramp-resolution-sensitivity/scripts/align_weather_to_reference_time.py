#!/usr/bin/env python3
"""Align gridded weather NetCDF variables to a reference time coordinate."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr


WEATHER_VARS = ["DSWRFsfc", "t2mz", "UGRD10m", "VGRD10m", "u100m", "v100m"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reference-var", default="DSWRFsfc")
    parser.add_argument("--vars", nargs="+", default=WEATHER_VARS)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    reference_root = Path(args.reference_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    reference_path = reference_root / f"{args.reference_var}.nc"
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    with xr.open_dataset(reference_path) as ref:
        target_time = ref["time"].values
        target_time_pd = pd.to_datetime(target_time)

    rows = []
    for var in args.vars:
        src_path = input_root / f"{var}.nc"
        if not src_path.exists():
            rows.append({"variable": var, "status": "missing", "input_time_steps": 0, "output_time_steps": 0})
            continue

        with xr.open_dataset(src_path) as ds:
            input_times = pd.to_datetime(ds["time"].values)
            aligned = ds.reindex(time=target_time)
            missing_after = int(aligned[var].isnull().sum().values) if var in aligned else int(next(iter(aligned.data_vars.values())).isnull().sum().values)
            aligned.to_netcdf(output_root / f"{var}.nc")
            rows.append(
                {
                    "variable": var,
                    "status": "ok",
                    "input_time_steps": len(input_times),
                    "input_time_start": str(input_times.min()),
                    "input_time_end": str(input_times.max()),
                    "output_time_steps": len(target_time_pd),
                    "output_time_start": str(target_time_pd.min()),
                    "output_time_end": str(target_time_pd.max()),
                    "missing_values_after_alignment": missing_after,
                }
            )
            print(f"[OK] wrote {output_root / f'{var}.nc'}")

    summary = pd.DataFrame(rows)
    lines = [
        "# Weather reference-time alignment",
        "",
        f"- Input root: {input_root}",
        f"- Reference root: {reference_root}",
        f"- Output root: {output_root}",
        f"- Reference time steps: {len(target_time_pd)}",
        f"- Reference time start: {target_time_pd.min()}",
        f"- Reference time end: {target_time_pd.max()}",
        "",
        "## Variable Alignment",
        summary.to_markdown(index=False),
    ]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {report}")


if __name__ == "__main__":
    main()
