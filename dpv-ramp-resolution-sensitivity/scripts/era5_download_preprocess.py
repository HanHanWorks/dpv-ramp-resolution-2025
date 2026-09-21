#!/usr/bin/env python3
"""Download and standardize ERA5 single-level data for DPV ramp comparisons."""

from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ERA5_VARIABLES = {
    "surface_solar_radiation_downwards": "ssrd",
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
}

STANDARD_NAMES = {
    "ssrd": "DSWRFsfc",
    "t2m": "t2mz",
    "u10": "UGRD10m",
    "v10": "VGRD10m",
    "u100": "u100m",
    "v100": "v100m",
}

ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_VARIABLES = {
    "surface_solar_radiation_downwards": "DSWRFsfc",
    "2m_temperature": "t2mz",
    "10m_u_component_of_wind": "UGRD10m",
    "10m_v_component_of_wind": "VGRD10m",
    "100m_u_component_of_wind": "u100m",
    "100m_v_component_of_wind": "v100m",
}


def month_days(month: str) -> list[str]:
    year, mon = [int(x) for x in month.split("-")]
    return [f"{day:02d}" for day in range(1, calendar.monthrange(year, mon)[1] + 1)]


def build_request(month: str, area: list[float]) -> dict[str, object]:
    year, mon = month.split("-")
    return {
        "product_type": ["reanalysis"],
        "variable": list(ERA5_VARIABLES),
        "year": [year],
        "month": [mon],
        "day": month_days(month),
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def write_request(month: str, output: Path, area: list[float]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = build_request(month, area)
    output.write_text(json.dumps({"dataset": "reanalysis-era5-single-levels", "request": request}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote request manifest {output}")


def download(month: str, target: Path, area: list[float]) -> None:
    cdsapirc = Path.home() / ".cdsapirc"
    if not cdsapirc.exists():
        raise SystemExit(f"Missing {cdsapirc}; configure CDS API credentials before downloading ERA5.")
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit("Missing cdsapi package; install it in the active environment.") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-single-levels", build_request(month, area), str(target))
    print(f"[OK] wrote {target}")


def coord_name(ds: xr.Dataset, names: list[str]) -> str:
    for name in names:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"Cannot find coordinate among {names}; coords={list(ds.coords)}, dims={list(ds.dims)}")


def time_name(ds: xr.Dataset) -> str:
    for name in ["valid_time", "time"]:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"Cannot find time coordinate; coords={list(ds.coords)}, dims={list(ds.dims)}")


def standardize_da(ds: xr.Dataset, short_name: str) -> xr.DataArray:
    if short_name not in ds:
        candidates = [name for name in ds.data_vars if name.lower() == short_name.lower()]
        if not candidates:
            raise KeyError(f"Cannot find ERA5 variable {short_name}; data_vars={list(ds.data_vars)}")
        short_name = candidates[0]

    da = ds[short_name]
    tname = time_name(ds)
    lon_name = coord_name(ds, ["longitude", "lon"])
    lat_name = coord_name(ds, ["latitude", "lat"])
    da = da.rename({tname: "time", lon_name: "grid_xt", lat_name: "grid_yt"})
    da = da.sortby("grid_yt").sortby("grid_xt").sortby("time")
    da = da.transpose("time", "grid_yt", "grid_xt")

    standard_name = STANDARD_NAMES[short_name]
    if short_name == "ssrd":
        units = str(da.attrs.get("units", "")).lower()
        if "j" in units or float(da.max(skipna=True).values) > 2000:
            da = da / 3600.0
        da = da.clip(min=0)
        da.attrs["units"] = "W m-2"
        da.attrs["conversion_note"] = "ERA5 ssrd accumulation converted to hourly mean irradiance by dividing J m-2 by 3600 s."
    da = da.rename(standard_name)
    return da.astype("float32")


def preprocess(raw: Path, output_root: Path, month: str) -> None:
    if not raw.exists():
        raise FileNotFoundError(raw)
    output_dir = output_root / month
    output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(raw) as ds:
        for short_name, standard_name in STANDARD_NAMES.items():
            da = standardize_da(ds, short_name)
            out = xr.Dataset({standard_name: da})
            out.attrs["source"] = "ERA5 single levels reanalysis"
            out.attrs["month"] = month
            path = output_dir / f"{standard_name}.nc"
            out.to_netcdf(path)
            print(f"[OK] wrote {path}")


def download_arco(month: str, output_root: Path, area: list[float], store: str = ARCO_STORE) -> None:
    """Read a small regional subset from the public ARCO ERA5 Zarr store.

    The ARCO store is anonymous and covers recent ERA5/ERA5T data, so it is a
    useful fallback when local CDS credentials are not configured. Chunks are
    global per hour; this should be used only for small month/region extracts.
    """
    try:
        import gcsfs
    except ImportError as exc:
        raise SystemExit("Missing gcsfs/zarr dependencies; install gcsfs zarr fsspec dask first.") from exc

    year, mon = [int(part) for part in month.split("-")]
    start = pd.Timestamp(year=year, month=mon, day=1, hour=0)
    # Include the first hour of the next month so ERA5 can be reindexed to the
    # Tianji daily-run convention (01:00 through next-day 00:00) without an
    # all-NaN terminal hour.
    end = pd.Timestamp(year=year, month=mon, day=calendar.monthrange(year, mon)[1], hour=23) + pd.Timedelta(hours=1)
    north, west, south, east = area
    output_dir = output_root / month
    output_dir.mkdir(parents=True, exist_ok=True)

    mapper = gcsfs.GCSFileSystem(token="anon").get_mapper(store)
    ds = xr.open_zarr(mapper, consolidated=True, chunks={})
    subset = ds[list(ARCO_VARIABLES)].sel(
        time=slice(start.to_datetime64(), end.to_datetime64()),
        latitude=slice(north, south),
        longitude=slice(west, east),
    )

    for era5_name, standard_name in ARCO_VARIABLES.items():
        da = subset[era5_name].rename({"latitude": "grid_yt", "longitude": "grid_xt"})
        da = da.sortby("grid_yt").sortby("grid_xt").sortby("time")
        da = da.transpose("time", "grid_yt", "grid_xt")
        if era5_name == "surface_solar_radiation_downwards":
            da = (da / 3600.0).clip(min=0)
            da.attrs["units"] = "W m-2"
            da.attrs["conversion_note"] = "ARCO ERA5 ssrd accumulation converted from J m-2 to hourly mean irradiance by dividing by 3600 s."
        da = da.rename(standard_name).astype("float32")
        out = xr.Dataset({standard_name: da})
        out.attrs["source"] = "ARCO ERA5 public Zarr on Google Cloud"
        out.attrs["month"] = month
        out.attrs["area_north_west_south_east"] = ",".join(str(v) for v in area)
        path = output_dir / f"{standard_name}.nc"
        out.to_netcdf(path)
        print(f"[OK] wrote {path}")


def map_to_tianji(era5_root: Path, tianji_root: Path, output_root: Path, month: str) -> None:
    output_dir = output_root / month
    output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(tianji_root / month / "DSWRFsfc.nc") as tmpl:
        target_lon = tmpl["grid_xt"]
        target_lat = tmpl["grid_yt"]
        target_time = tmpl["time"]

    for standard_name in STANDARD_NAMES.values():
        src_path = era5_root / month / f"{standard_name}.nc"
        if not src_path.exists():
            print(f"[WARN] missing {src_path}")
            continue
        with xr.open_dataset(src_path) as ds:
            da = ds[standard_name]
            mapped = da.interp(grid_xt=target_lon, grid_yt=target_lat, method="linear")
            mapped = mapped.reindex(time=target_time, method="nearest", tolerance=np.timedelta64(30, "m"))
            mapped = mapped.rename(standard_name).astype("float32")
            out = xr.Dataset({standard_name: mapped})
            out.attrs["source"] = "ERA5 mapped to Tianji grid by bilinear spatial interpolation and nearest hourly time alignment"
            path = output_dir / f"{standard_name}.nc"
            out.to_netcdf(path)
            print(f"[OK] wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common_area = [32.25, 108.75, 29.25, 112.25]
    req = sub.add_parser("write-request")
    req.add_argument("--month", required=True)
    req.add_argument("--output", required=True)
    req.add_argument("--area", nargs=4, type=float, default=common_area, help="North West South East")

    dl = sub.add_parser("download")
    dl.add_argument("--month", required=True)
    dl.add_argument("--target", required=True)
    dl.add_argument("--area", nargs=4, type=float, default=common_area, help="North West South East")

    arco = sub.add_parser("download-arco", help="Download a regional subset from public ARCO ERA5 Zarr")
    arco.add_argument("--month", required=True)
    arco.add_argument("--output-root", default="data/processed/era5_monthly")
    arco.add_argument("--area", nargs=4, type=float, default=common_area, help="North West South East")
    arco.add_argument("--store", default=ARCO_STORE)

    prep = sub.add_parser("preprocess")
    prep.add_argument("--raw", required=True)
    prep.add_argument("--month", required=True)
    prep.add_argument("--output-root", default="data/processed/era5_monthly")

    mapper = sub.add_parser("map-to-tianji")
    mapper.add_argument("--month", required=True)
    mapper.add_argument("--era5-root", default="data/processed/era5_monthly")
    mapper.add_argument("--tianji-root", default="data/processed/tianji_monthly")
    mapper.add_argument("--output-root", default="data/processed/era5_on_tianji_monthly")

    args = parser.parse_args()
    if args.command == "write-request":
        write_request(args.month, Path(args.output), args.area)
    elif args.command == "download":
        download(args.month, Path(args.target), args.area)
    elif args.command == "download-arco":
        download_arco(args.month, Path(args.output_root), args.area, args.store)
    elif args.command == "preprocess":
        preprocess(Path(args.raw), Path(args.output_root), args.month)
    elif args.command == "map-to-tianji":
        map_to_tianji(Path(args.era5_root), Path(args.tianji_root), Path(args.output_root), args.month)


if __name__ == "__main__":
    main()
