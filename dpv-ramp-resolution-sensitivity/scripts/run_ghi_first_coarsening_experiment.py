#!/usr/bin/env python3
"""Run weather-field-first coarsening before PV conversion.

This experiment complements the PV-field coarsening experiment used in the
associated SETA manuscript. It first coarsens Tianji GHI, temperature, and wind
fields, then applies the same simplified PV conversion and ramp workflow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from run_ramp_resolution_experiments import (
    OUT,
    RampProduct,
    cluster_events,
    detect_events,
    product_summary,
    spatial_structure,
    terrain_summary,
)


ROOT = Path(".")


def coarsen_and_regrid_weather(da: xr.DataArray, factor: int, method: str = "nearest") -> xr.DataArray:
    if factor == 1:
        return da.copy()
    coarse = da.coarsen(grid_yt=factor, grid_xt=factor, boundary="pad").mean()
    return coarse.interp(
        grid_yt=da.grid_yt,
        grid_xt=da.grid_xt,
        method=method,
        kwargs={"fill_value": "extrapolate"},
    )


def pv_from_weather_fields(weather_root: Path, factor: int, method: str = "nearest") -> xr.DataArray:
    with (
        xr.open_dataset(weather_root / "DSWRFsfc.nc") as ghi_ds,
        xr.open_dataset(weather_root / "t2mz.nc") as t_ds,
        xr.open_dataset(weather_root / "UGRD10m.nc") as u_ds,
        xr.open_dataset(weather_root / "VGRD10m.nc") as v_ds,
    ):
        ghi = ghi_ds["DSWRFsfc"].clip(min=0).load()
        tamb_c = (t_ds["t2mz"] - 273.15).load()
        u10 = u_ds["UGRD10m"].load()
        v10 = v_ds["VGRD10m"].load()

    ghi_c = coarsen_and_regrid_weather(ghi, factor, method).clip(min=0)
    tamb_c = coarsen_and_regrid_weather(tamb_c, factor, method)
    u10 = coarsen_and_regrid_weather(u10, factor, method)
    v10 = coarsen_and_regrid_weather(v10, factor, method)
    ws10m = np.sqrt(u10**2 + v10**2)
    tcell = tamb_c + ghi_c / (26.9 + 6.2 * ws10m)
    pv_pu = (ghi_c / 1000.0) * (1 - 0.0035 * (tcell - 25.0)) * 0.96
    return pv_pu.clip(min=0, max=1.2).rename("pv_pu").astype("float32")


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    terrain = pd.read_csv(ROOT / "data/processed/terrain/terrain_classes.csv")
    weather_root = ROOT / "data/processed/tianji_monthly/2025-07"
    products = [
        RampProduct("Tianji_weather_2.5km", pv_from_weather_fields(weather_root, 1), 2.5),
        RampProduct("Tianji_weather_coarsened_5km", pv_from_weather_fields(weather_root, 2), 5.0),
        RampProduct("Tianji_weather_coarsened_10km", pv_from_weather_fields(weather_root, 4), 10.0),
        RampProduct("Tianji_weather_coarsened_25km", pv_from_weather_fields(weather_root, 10), 25.0),
        RampProduct("Tianji_weather_coarsened_0.25deg_like", pv_from_weather_fields(weather_root, 11), 27.5),
    ]

    summary_rows = []
    terrain_frames = []
    threshold = 0.3
    for product in products:
        events = detect_events(product.pv, threshold)
        _, clusters = cluster_events(events)
        summary_rows.append(product_summary(product.label, product.nominal_resolution_km, product.pv, events, clusters, threshold))
        terrain_frames.append(terrain_summary(product.label, product.nominal_resolution_km, events, terrain, threshold))
        print(f"[OK] weather-first coarsening processed {product.label}")

    weather_first = pd.DataFrame(summary_rows)
    terrain_summary_df = pd.concat(terrain_frames, ignore_index=True)

    pv_first_path = OUT / "controlled_tianji_coarsening_summary.csv"
    comparison = pd.DataFrame()
    if pv_first_path.exists():
        pv_first = pd.read_csv(pv_first_path)
        pv_first = pv_first[pv_first["product"].str.startswith("Tianji")].copy()
        pv_first["coarsening_pathway"] = "PV-field first"
        weather_first_for_compare = weather_first.copy()
        weather_first_for_compare["coarsening_pathway"] = "Weather-field first"
        comparison = pd.concat([pv_first, weather_first_for_compare], ignore_index=True)

    return weather_first, terrain_summary_df, comparison


def run_regridding_sensitivity() -> pd.DataFrame:
    weather_root = ROOT / "data/processed/tianji_monthly/2025-07"
    product_specs = [
        ("2.5km", 1, 2.5),
        ("5km", 2, 5.0),
        ("10km", 4, 10.0),
        ("25km", 10, 25.0),
        ("0.25deg_like", 11, 27.5),
    ]
    rows = []
    threshold = 0.3
    for method in ["nearest", "linear"]:
        for suffix, factor, resolution in product_specs:
            pv = pv_from_weather_fields(weather_root, factor, method=method)
            events = detect_events(pv, threshold)
            _, clusters = cluster_events(events)
            row = product_summary(
                f"Tianji_weather_{method}_{suffix}",
                resolution,
                pv,
                events,
                clusters,
                threshold,
            )
            row["regridding_method"] = method
            rows.append(row)
            print(f"[OK] regridding sensitivity processed {method} {suffix}")
    return pd.DataFrame(rows)


def run_weather_first_threshold_and_structure() -> tuple[pd.DataFrame, pd.DataFrame]:
    weather_root = ROOT / "data/processed/tianji_monthly/2025-07"
    product_specs = [
        ("Tianji_weather_2.5km", 1, 2.5),
        ("Tianji_weather_coarsened_5km", 2, 5.0),
        ("Tianji_weather_coarsened_10km", 4, 10.0),
        ("Tianji_weather_coarsened_25km", 10, 25.0),
        ("Tianji_weather_coarsened_0.25deg_like", 11, 27.5),
    ]
    threshold_rows = []
    structure_rows = []
    for label, factor, resolution in product_specs:
        pv = pv_from_weather_fields(weather_root, factor, method="nearest")
        structure_rows.append(spatial_structure(label, resolution, pv))
        for threshold in [0.2, 0.3, 0.4]:
            events = detect_events(pv, threshold)
            _, clusters = cluster_events(events)
            threshold_rows.append(product_summary(label, resolution, pv, events, clusters, threshold))
            print(f"[OK] weather-first threshold processed {label} threshold={threshold}")

    structure = pd.DataFrame(structure_rows)
    base_lag1 = structure.loc[
        structure["product"].eq("Tianji_weather_2.5km"),
        "semivariance_lag1_grid",
    ].iloc[0]
    structure["short_scale_variance_retention_vs_tianji"] = structure["semivariance_lag1_grid"] / base_lag1
    return pd.DataFrame(threshold_rows), structure


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary, terrain, comparison = run_experiment()
    regridding = run_regridding_sensitivity()
    threshold, structure = run_weather_first_threshold_and_structure()
    summary.to_csv(OUT / "ghi_first_coarsening_summary.csv", index=False)
    terrain.to_csv(OUT / "ghi_first_coarsening_terrain_summary.csv", index=False)
    regridding.to_csv(OUT / "weather_regridding_sensitivity.csv", index=False)
    threshold.to_csv(OUT / "ghi_first_threshold_sensitivity.csv", index=False)
    structure.to_csv(OUT / "ghi_first_spatial_structure_summary.csv", index=False)
    if not comparison.empty:
        comparison.to_csv(OUT / "coarsening_pathway_comparison.csv", index=False)
    print(f"[OK] wrote {OUT / 'ghi_first_coarsening_summary.csv'}")
    print(f"[OK] wrote {OUT / 'ghi_first_coarsening_terrain_summary.csv'}")
    print(f"[OK] wrote {OUT / 'weather_regridding_sensitivity.csv'}")
    print(f"[OK] wrote {OUT / 'ghi_first_threshold_sensitivity.csv'}")
    print(f"[OK] wrote {OUT / 'ghi_first_spatial_structure_summary.csv'}")
    if not comparison.empty:
        print(f"[OK] wrote {OUT / 'coarsening_pathway_comparison.csv'}")


if __name__ == "__main__":
    main()
