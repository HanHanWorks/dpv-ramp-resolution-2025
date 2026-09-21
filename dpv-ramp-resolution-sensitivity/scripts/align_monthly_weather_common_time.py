#!/usr/bin/env python3
"""Align monthly weather NetCDF variables to their common time axis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


WEATHER_VARS = ["DSWRFsfc", "t2mz", "UGRD10m", "VGRD10m", "u100m", "v100m"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--vars", nargs="+", default=WEATHER_VARS)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    time_sets = {}
    rows = []
    for var in args.vars:
        path = input_root / f"{var}.nc"
        if not path.exists():
            rows.append({"variable": var, "status": "missing", "time_steps": 0})
            continue
        with xr.open_dataset(path) as ds:
            times = pd.to_datetime(ds["time"].values)
            time_sets[var] = set(times)
            rows.append({"variable": var, "status": "ok", "time_steps": len(times), "time_start": str(times.min()), "time_end": str(times.max())})
    if len(time_sets) != len(args.vars):
        raise SystemExit("Cannot align month: one or more variables are missing.")
    common = sorted(set.intersection(*time_sets.values()))
    if not common:
        raise SystemExit("Cannot align month: variables have no common timestamps.")
    common_np = np.array(common, dtype="datetime64[ns]")
    for var in args.vars:
        with xr.open_dataset(input_root / f"{var}.nc") as ds:
            aligned = ds.sel(time=common_np)
            aligned.to_netcdf(output_root / f"{var}.nc")
            print(f"[OK] wrote {output_root / f'{var}.nc'}")
    summary = pd.DataFrame(rows)
    lines = [
        "# Monthly common-time alignment",
        "",
        f"- Input root: {input_root}",
        f"- Output root: {output_root}",
        f"- Common time steps: {len(common)}",
        f"- Common time start: {common[0]}",
        f"- Common time end: {common[-1]}",
        "",
        "## Variable Time Coverage",
        summary.to_markdown(index=False),
    ]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {report}")


if __name__ == "__main__":
    main()
