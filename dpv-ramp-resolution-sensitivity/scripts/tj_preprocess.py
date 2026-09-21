#!/usr/bin/env python3
"""Preprocess Tianji TJ-CN Huazhong NetCDF files for DPV ramp studies.

The script is intentionally conservative: it first works with the official
directory layout and preserves raw files. It supports inspection, regional
cropping, monthly merging, and point extraction for CMA stations or feeder
nodes.
"""

from __future__ import annotations

import argparse
import calendar
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_VARS = ["DSWRFsfc", "t2mz", "UGRD10m", "VGRD10m", "u100m", "v100m"]


@dataclass(frozen=True)
class Bounds:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def month_dir(day: date) -> str:
    return f"{day.month:02d}"


def yyyymmddhh(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H")


def find_var_files(root: Path, start: date, end: date, var: str) -> list[Path]:
    """Find files in the official layout and common standard-download layouts."""
    files: list[Path] = []

    for day in iter_days(start, end):
        month = month_dir(day)
        stamp = day.strftime("%Y%m%d00")

        # Official historical layout:
        # root/{MM}/{base_yyyymmddhh}/{valid_yyyymmddhh}/{var}.nc
        official = root / month / stamp / stamp / f"{var}.nc"
        if official.exists():
            files.append(official)

        # Standard-download examples sometimes group by variable:
        # root/{var}/{base}_{var}_f*.nc
        files.extend(sorted((root / var).glob(f"{stamp}_{var}_f*.nc")))

        # Flat fallback:
        files.extend(sorted(root.glob(f"**/{stamp}*/{stamp}/{var}.nc")))

    unique = sorted(set(path for path in files if path.is_file()))
    return unique


def get_coord_names(ds: xr.Dataset) -> tuple[str, str]:
    lon_candidates = ["grid_xt", "lon", "longitude", "x"]
    lat_candidates = ["grid_yt", "lat", "latitude", "y"]

    lon_name = next((name for name in lon_candidates if name in ds.coords or name in ds.dims), None)
    lat_name = next((name for name in lat_candidates if name in ds.coords or name in ds.dims), None)

    if lon_name is None or lat_name is None:
        raise ValueError(
            "Could not identify longitude/latitude coordinates. "
            f"coords={list(ds.coords)}, dims={list(ds.dims)}"
        )

    return lon_name, lat_name


def crop_dataset(ds: xr.Dataset, bounds: Bounds) -> xr.Dataset:
    lon_name, lat_name = get_coord_names(ds)

    lon_values = ds[lon_name].values
    lat_values = ds[lat_name].values

    lon_slice = slice(bounds.lon_min, bounds.lon_max)
    if lon_values[0] > lon_values[-1]:
        lon_slice = slice(bounds.lon_max, bounds.lon_min)

    lat_slice = slice(bounds.lat_min, bounds.lat_max)
    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(bounds.lat_max, bounds.lat_min)

    return ds.sel({lon_name: lon_slice, lat_name: lat_slice})


def infer_valid_time_from_path(path: Path) -> pd.Timestamp | None:
    for part in reversed(path.parts):
        if len(part) == 10 and part.isdigit():
            try:
                return pd.Timestamp(datetime.strptime(part, "%Y%m%d%H"), tz="UTC")
            except ValueError:
                continue
    return None


def ensure_time(ds: xr.Dataset, path: Path) -> xr.Dataset:
    if "time" in ds.coords or "time" in ds.dims:
        return ds

    valid_time = infer_valid_time_from_path(path)
    if valid_time is None:
        return ds

    return ds.expand_dims(time=[valid_time.to_datetime64()])


def normalize_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds
    return ds.sortby("time")


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "t2mz" in out:
        out["t2m_C"] = out["t2mz"] - 273.15
    if {"UGRD10m", "VGRD10m"}.issubset(out.columns):
        out["ws10m"] = np.sqrt(out["UGRD10m"] ** 2 + out["VGRD10m"] ** 2)
    if {"u100m", "v100m"}.issubset(out.columns):
        out["ws100m"] = np.sqrt(out["u100m"] ** 2 + out["v100m"] ** 2)
        out["wd100m_deg"] = (270 - np.degrees(np.arctan2(out["v100m"], out["u100m"]))) % 360
    return out


def inspect_file(args: argparse.Namespace) -> None:
    path = Path(args.file)
    with xr.open_dataset(path) as ds:
        print(f"file: {path}")
        print(ds)
        print("\nVariables:")
        for name, da in ds.data_vars.items():
            print(f"- {name}: dims={da.dims}, shape={da.shape}, attrs={dict(da.attrs)}")
        print("\nCoordinates:")
        for name, coord in ds.coords.items():
            print(f"- {name}: dims={coord.dims}, shape={coord.shape}, attrs={dict(coord.attrs)}")


def crop_files(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    bounds = Bounds(args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    start = parse_date(args.start)
    end = parse_date(args.end)

    for var in args.vars:
        files = find_var_files(input_root, start, end, var)
        if not files:
            print(f"[WARN] no files found for {var}")
            continue

        for src in files:
            rel = src.relative_to(input_root)
            dst = output_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with xr.open_dataset(src) as ds:
                ds = ensure_time(ds, src)
                cropped = crop_dataset(ds, bounds)
                cropped.to_netcdf(dst)
            print(f"[OK] cropped {src} -> {dst}")


def merge_monthly(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    start = parse_date(args.start)
    end = parse_date(args.end)

    month_key = f"{start.year:04d}-{start.month:02d}"
    output_dir = output_root / month_key
    output_dir.mkdir(parents=True, exist_ok=True)

    for var in args.vars:
        files = find_var_files(input_root, start, end, var)
        if not files:
            print(f"[WARN] no files found for {var}")
            continue

        datasets = []
        for path in files:
            ds = xr.open_dataset(path)
            ds = ensure_time(ds, path)
            datasets.append(ds)

        try:
            merged = xr.concat(datasets, dim="time") if "time" in datasets[0].dims else xr.merge(datasets)
            merged = normalize_time(merged)
            if "time" in merged.coords:
                _, index = np.unique(merged["time"].values, return_index=True)
                merged = merged.isel(time=np.sort(index))
            dst = output_dir / f"{var}.nc"
            merged.to_netcdf(dst)
            print(f"[OK] merged {len(files)} files -> {dst}")
        finally:
            for ds in datasets:
                ds.close()


def open_var_dataset(input_root: Path, var: str) -> xr.Dataset:
    path = input_root / f"{var}.nc"
    if not path.exists():
        matches = sorted(input_root.glob(f"**/{var}.nc"))
        if not matches:
            raise FileNotFoundError(f"Cannot find {var}.nc under {input_root}")
        path = matches[0]
    return xr.open_dataset(path)


def extract_points(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    points = pd.read_csv(args.points)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    required_cols = {"lon", "lat"}
    if not required_cols.issubset(points.columns):
        raise ValueError(f"points CSV must contain columns: {sorted(required_cols)}")

    id_col = "station_id" if "station_id" in points.columns else "node_id" if "node_id" in points.columns else None
    if id_col is None:
        points["point_id"] = [f"point_{i+1:03d}" for i in range(len(points))]
        id_col = "point_id"

    rows: list[pd.DataFrame] = []
    datasets: dict[str, xr.Dataset] = {}

    try:
        for var in args.vars:
            datasets[var] = open_var_dataset(input_root, var)

        first_ds = next(iter(datasets.values()))
        lon_name, lat_name = get_coord_names(first_ds)

        for _, point in points.iterrows():
            point_id = point[id_col]
            lon = float(point["lon"])
            lat = float(point["lat"])
            data: dict[str, np.ndarray] = {}
            time_values = None

            for var, ds in datasets.items():
                da = ds[var] if var in ds.data_vars else next(iter(ds.data_vars.values()))
                if args.method == "nearest":
                    sample = da.sel({lon_name: lon, lat_name: lat}, method="nearest")
                else:
                    sample = da.interp({lon_name: lon, lat_name: lat}, method="linear")

                data[var] = np.asarray(sample.values).reshape(-1)
                if "time" in sample.coords and time_values is None:
                    time_values = pd.to_datetime(sample["time"].values)

            length = max(len(values) for values in data.values())
            frame = pd.DataFrame({name: np.resize(values, length) for name, values in data.items()})
            if time_values is not None:
                time_series = pd.Series(pd.to_datetime(np.resize(time_values, length), utc=True))
                frame.insert(0, "time_UTC", time_series.dt.strftime("%Y-%m-%d %H:%M:%S%z"))
                frame.insert(1, "time_BJT", time_series.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d %H:%M:%S%z"))
            frame.insert(0, "point_id", point_id)
            frame.insert(1, "lon", lon)
            frame.insert(2, "lat", lat)
            frame = add_derived_columns(frame)
            rows.append(frame)

        result = pd.concat(rows, ignore_index=True)
        result.to_csv(output, index=False, encoding="utf-8-sig")
        print(f"[OK] wrote {output}")
    finally:
        for ds in datasets.values():
            ds.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Inspect one NetCDF file")
    inspect.add_argument("--file", required=True)
    inspect.set_defaults(func=inspect_file)

    crop = sub.add_parser("crop", help="Crop files to a lon/lat bounding box")
    crop.add_argument("--input-root", required=True)
    crop.add_argument("--output-root", required=True)
    crop.add_argument("--start", required=True)
    crop.add_argument("--end", required=True)
    crop.add_argument("--vars", nargs="+", default=DEFAULT_VARS)
    crop.add_argument("--lon-min", type=float, required=True)
    crop.add_argument("--lon-max", type=float, required=True)
    crop.add_argument("--lat-min", type=float, required=True)
    crop.add_argument("--lat-max", type=float, required=True)
    crop.set_defaults(func=crop_files)

    merge = sub.add_parser("merge", help="Merge cropped files by variable")
    merge.add_argument("--input-root", required=True)
    merge.add_argument("--output-root", required=True)
    merge.add_argument("--start", required=True)
    merge.add_argument("--end", required=True)
    merge.add_argument("--vars", nargs="+", default=DEFAULT_VARS)
    merge.set_defaults(func=merge_monthly)

    extract = sub.add_parser("extract-points", help="Extract station or feeder-node time series")
    extract.add_argument("--input-root", required=True)
    extract.add_argument("--points", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--vars", nargs="+", default=DEFAULT_VARS)
    extract.add_argument("--method", choices=["nearest", "linear"], default="nearest")
    extract.set_defaults(func=extract_points)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
