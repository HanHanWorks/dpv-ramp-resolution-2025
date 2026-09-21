#!/usr/bin/env python3
"""Prepare terrain metrics on the Tianji analysis grid from Terrarium tiles."""

from __future__ import annotations

import argparse
import math
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from PIL import Image
from scipy.ndimage import generic_filter, sobel


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int, float, float]:
    lat_rad = math.radians(lat)
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return int(math.floor(x)), int(math.floor(y)), x, y


def fetch_tile(url_template: str, z: int, x: int, y: int, cache_dir: Path, timeout: int) -> np.ndarray:
    cache_path = cache_dir / str(z) / str(x) / f"{y}.png"
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        url = url_template.format(z=z, x=x, y=y)
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "dpv-ramp-terrain/0.1"})
        response.raise_for_status()
        cache_path.write_bytes(response.content)
    image = Image.open(BytesIO(cache_path.read_bytes())).convert("RGB")
    arr = np.asarray(image).astype("float32")
    elevation = (arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0) - 32768.0
    return elevation


def sample_terrarium(
    lons: np.ndarray,
    lats: np.ndarray,
    zoom: int,
    cache_dir: Path,
    url_template: str,
    timeout: int,
) -> np.ndarray:
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())
    x_min, y_max, _, _ = lonlat_to_tile(lon_min, lat_min, zoom)
    x_max, y_min, _, _ = lonlat_to_tile(lon_max, lat_max, zoom)
    xs = range(min(x_min, x_max), max(x_min, x_max) + 1)
    ys = range(min(y_min, y_max), max(y_min, y_max) + 1)

    tiles: dict[tuple[int, int], np.ndarray] = {}
    for x in xs:
        for y in ys:
            tiles[(x, y)] = fetch_tile(url_template, zoom, x, y, cache_dir, timeout)

    out = np.empty((len(lats), len(lons)), dtype="float32")
    for iy, lat in enumerate(lats):
        for ix, lon in enumerate(lons):
            tx, ty, xf, yf = lonlat_to_tile(float(lon), float(lat), zoom)
            tile = tiles[(tx, ty)]
            px = (xf - tx) * 256.0
            py = (yf - ty) * 256.0
            x0 = int(np.clip(np.floor(px), 0, 255))
            y0 = int(np.clip(np.floor(py), 0, 255))
            x1 = min(x0 + 1, 255)
            y1 = min(y0 + 1, 255)
            wx = px - x0
            wy = py - y0
            out[iy, ix] = (
                tile[y0, x0] * (1 - wx) * (1 - wy)
                + tile[y0, x1] * wx * (1 - wy)
                + tile[y1, x0] * (1 - wx) * wy
                + tile[y1, x1] * wx * wy
            )
    return out


def terrain_metrics(elevation: np.ndarray, lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_lat = np.deg2rad(float(np.mean(lats)))
    dx_m = float(np.mean(np.diff(lons)) * 111_000.0 * np.cos(mean_lat))
    dy_m = float(np.mean(np.diff(lats)) * 111_000.0)
    dzdx = sobel(elevation, axis=1, mode="nearest") / (8.0 * dx_m)
    dzdy = sobel(elevation, axis=0, mode="nearest") / (8.0 * dy_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype("float32")

    def tri_func(values: np.ndarray) -> float:
        center = values[len(values) // 2]
        return float(np.sqrt(np.mean((values - center) ** 2)))

    tri = generic_filter(elevation, tri_func, size=3, mode="nearest").astype("float32")
    relief = generic_filter(elevation, lambda values: float(np.max(values) - np.min(values)), size=5, mode="nearest").astype("float32")
    return slope_deg, tri, relief


def classify_terrain(elevation: np.ndarray, tri: np.ndarray) -> np.ndarray:
    elev_median = np.nanmedian(elevation)
    tri_median = np.nanmedian(tri)
    classes = np.empty(elevation.shape, dtype=object)
    classes[(elevation < elev_median) & (tri < tri_median)] = "low_relief_low_elevation"
    classes[(elevation >= elev_median) & (tri < tri_median)] = "low_relief_high_elevation"
    classes[(elevation < elev_median) & (tri >= tri_median)] = "high_relief_mid_elevation"
    classes[(elevation >= elev_median) & (tri >= tri_median)] = "high_relief_high_elevation"
    return classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-grid", default="data/processed/pv_grid/2025-07/pv_pu.nc")
    parser.add_argument("--output-nc", default="data/processed/terrain/terrain_on_tianji_grid.nc")
    parser.add_argument("--classes-csv", default="data/processed/terrain/terrain_classes.csv")
    parser.add_argument("--cache-dir", default="data/raw/terrain/terrarium_tiles")
    parser.add_argument("--zoom", type=int, default=10)
    parser.add_argument("--tile-url", default="https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    with xr.open_dataset(args.template_grid) as template:
        lons = template["grid_xt"].values.astype("float64")
        lats = template["grid_yt"].values.astype("float64")

    elevation = sample_terrarium(lons, lats, args.zoom, Path(args.cache_dir), args.tile_url, args.timeout)
    slope, tri, relief = terrain_metrics(elevation, lons, lats)
    classes = classify_terrain(elevation, tri)

    class_names = sorted(pd.unique(classes.ravel()))
    class_to_id = {name: i + 1 for i, name in enumerate(class_names)}
    class_id = np.vectorize(class_to_id.get)(classes).astype("int16")

    ds = xr.Dataset(
        {
            "elevation_m": (("grid_yt", "grid_xt"), elevation.astype("float32")),
            "slope_deg": (("grid_yt", "grid_xt"), slope),
            "tri_m": (("grid_yt", "grid_xt"), tri),
            "relief_5x5_m": (("grid_yt", "grid_xt"), relief),
            "terrain_class_id": (("grid_yt", "grid_xt"), class_id),
        },
        coords={"grid_yt": lats.astype("float32"), "grid_xt": lons.astype("float32")},
        attrs={
            "description": "Terrain metrics sampled to the Tianji Phase 1 core-region grid",
            "source": "AWS Terrain Tiles Terrarium elevation tiles",
            "tile_url_template": args.tile_url,
            "zoom": args.zoom,
            "classification": "Median elevation x median TRI four-class split",
        },
    )
    ds["terrain_class_id"].attrs["class_mapping"] = "; ".join(f"{v}={k}" for k, v in class_to_id.items())

    output_nc = Path(args.output_nc)
    output_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_nc)
    print(f"[OK] wrote {output_nc}")

    rows = []
    for iy, lat in enumerate(lats):
        for ix, lon in enumerate(lons):
            rows.append(
                {
                    "grid_x": ix,
                    "grid_y": iy,
                    "lon": float(lon),
                    "lat": float(lat),
                    "elevation_m": float(elevation[iy, ix]),
                    "slope_deg": float(slope[iy, ix]),
                    "tri_m": float(tri[iy, ix]),
                    "relief_5x5_m": float(relief[iy, ix]),
                    "terrain_class_id": int(class_id[iy, ix]),
                    "terrain_class": str(classes[iy, ix]),
                }
            )
    classes_csv = Path(args.classes_csv)
    classes_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(classes_csv, index=False, encoding="utf-8")
    print(f"[OK] wrote {classes_csv}")
    print(pd.Series(classes.ravel()).value_counts().to_string())


if __name__ == "__main__":
    main()
