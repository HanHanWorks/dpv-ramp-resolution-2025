#!/usr/bin/env python3
"""Build DPV ramp resolution-sensitivity experiment tables and figure drafts.

The outputs are supporting evidence for the associated SETA manuscript.
They are not an observational calibration workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import DBSCAN


ROOT = Path(".")
OUT = ROOT / "data/analysis/seta_experiments"
FIG = ROOT / "figures/seta_experiments"
PYTHON = "python3"


@dataclass
class RampProduct:
    label: str
    pv: xr.DataArray
    nominal_resolution_km: float


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()
    print(f"[OK] wrote {path}")


def open_pv(path: Path) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        return ds["pv_pu"].load()


def coarsen_and_regrid(pv: xr.DataArray, factor: int) -> xr.DataArray:
    if factor == 1:
        return pv.copy()
    coarse = pv.coarsen(grid_yt=factor, grid_xt=factor, boundary="trim").mean()
    return coarse.interp(grid_yt=pv.grid_yt, grid_xt=pv.grid_xt, method="nearest").fillna(0).clip(0, 1.2)


def detect_events(pv: xr.DataArray, threshold: float, min_pv: float = 0.1, max_simultaneity: float = 0.8) -> pd.DataFrame:
    delta = pv.diff("time")
    p_prev = pv.isel(time=slice(None, -1))
    p_now = pv.isel(time=slice(1, None))
    valid = np.maximum(p_prev, p_now) >= min_pv
    mask = ((delta >= threshold) | (delta <= -threshold)) & valid
    if max_simultaneity < 1:
        mask = mask & (mask.mean(dim=("grid_yt", "grid_xt")) <= max_simultaneity)
    flat = mask.stack(event_id=("time", "grid_yt", "grid_xt"))
    idx = np.flatnonzero(flat.values)
    columns = ["time", "lon", "lat", "grid_x", "grid_y", "p_pu", "delta_p", "ramp_type", "threshold"]
    if len(idx) == 0:
        return pd.DataFrame(columns=columns)
    stacked = xr.Dataset({"p_pu": p_now, "delta_p": delta}).stack(event_id=("time", "grid_yt", "grid_xt")).isel(event_id=idx)
    df = stacked.to_dataframe().reset_index().rename(columns={"grid_xt": "lon", "grid_yt": "lat"})
    lon_to_ix = {round(float(v), 10): i for i, v in enumerate(pv.grid_xt.values)}
    lat_to_iy = {round(float(v), 10): i for i, v in enumerate(pv.grid_yt.values)}
    df["grid_x"] = df["lon"].map(lambda x: lon_to_ix[round(float(x), 10)])
    df["grid_y"] = df["lat"].map(lambda y: lat_to_iy[round(float(y), 10)])
    df["ramp_type"] = np.where(df["delta_p"] >= threshold, "up", "down")
    df["threshold"] = threshold
    df["time"] = pd.to_datetime(df["time"])
    return df[columns]


def cluster_events(events: pd.DataFrame, eps_space_km: float = 5.0, eps_time_h: float = 1.0, min_samples: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = events.copy()
    events["cluster_id"] = -1
    if events.empty:
        return events, pd.DataFrame()
    next_cluster_id = 0
    for _, group in events.groupby("ramp_type", sort=True):
        t0 = pd.to_datetime(group["time"]).min()
        time_h = (pd.to_datetime(group["time"]) - t0).dt.total_seconds().to_numpy() / 3600.0
        mean_lat = np.deg2rad(group["lat"].mean())
        x_km = group["lon"].to_numpy() * 111.0 * np.cos(mean_lat)
        y_km = group["lat"].to_numpy() * 111.0
        features = np.column_stack([x_km / eps_space_km, y_km / eps_space_km, time_h / eps_time_h])
        labels = DBSCAN(eps=1.0, min_samples=min_samples, n_jobs=-1).fit_predict(features)
        mapped = np.where(labels >= 0, labels + next_cluster_id, -1)
        events.loc[group.index, "cluster_id"] = mapped
        if labels.max(initial=-1) >= 0:
            next_cluster_id += int(labels.max()) + 1
    rows = []
    for cluster_id, group in events[events["cluster_id"] >= 0].groupby("cluster_id"):
        lons = group["lon"].to_numpy()
        lats = group["lat"].to_numpy()
        times = pd.to_datetime(group["time"])
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "event_count": len(group),
                "ramp_type_mode": group["ramp_type"].mode().iloc[0],
                "duration_h": (times.max() - times.min()).total_seconds() / 3600.0 + 1.0,
                "approx_area_km2": max(1.0, (lons.max() - lons.min()) * 111.0 * np.cos(np.deg2rad(lats.mean())))
                * max(1.0, (lats.max() - lats.min()) * 111.0),
                "peak_abs_delta_p": float(group["delta_p"].abs().max()),
            }
        )
    return events, pd.DataFrame(rows)


def product_summary(label: str, resolution_km: float, pv: xr.DataArray, events: pd.DataFrame, clusters: pd.DataFrame, threshold: float) -> dict[str, float | str]:
    n_grid = int(pv.sizes["grid_yt"] * pv.sizes["grid_xt"])
    n_time = int(pv.sizes["time"])
    sim = events.groupby("time").apply(lambda g: g[["grid_x", "grid_y"]].drop_duplicates().shape[0], include_groups=False) / n_grid if not events.empty else pd.Series(dtype=float)
    total_area = n_grid * 2.5 * 2.5
    return {
        "product": label,
        "nominal_resolution_km": resolution_km,
        "threshold": threshold,
        "time_steps": n_time,
        "grid_cells": n_grid,
        "event_count": int(len(events)),
        "event_rate_per_grid_hour": len(events) / max(1, n_grid * max(1, n_time - 1)),
        "up_events": int(events["ramp_type"].eq("up").sum()) if not events.empty else 0,
        "down_events": int(events["ramp_type"].eq("down").sum()) if not events.empty else 0,
        "up_share": float(events["ramp_type"].eq("up").mean()) if not events.empty else np.nan,
        "cluster_count": int(len(clusters)),
        "median_cluster_area_km2": float(clusters["approx_area_km2"].median()) if not clusters.empty else np.nan,
        "mean_cluster_area_km2": float(clusters["approx_area_km2"].mean()) if not clusters.empty else np.nan,
        "top_cluster_area_share": float(clusters["approx_area_km2"].max() / total_area) if not clusters.empty else np.nan,
        "p95_simultaneity": float(sim.quantile(0.95)) if not sim.empty else np.nan,
    }


def terrain_summary(label: str, resolution_km: float, events: pd.DataFrame, terrain: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    merged = events.merge(terrain[["grid_x", "grid_y", "terrain_class"]], on=["grid_x", "grid_y"], how="left")
    denominators = terrain.groupby("terrain_class").size()
    rows = []
    for terrain_class, group in merged.groupby("terrain_class"):
        rows.append(
            {
                "product": label,
                "nominal_resolution_km": resolution_km,
                "threshold": threshold,
                "terrain_class": terrain_class,
                "event_count": int(len(group)),
                "events_per_grid_cell": float(len(group) / denominators.get(terrain_class, 1)),
                "up_share": float(group["ramp_type"].eq("up").mean()),
            }
        )
    return pd.DataFrame(rows)


def nearest_neighbor_corr(delta: xr.DataArray, sample_times: int = 96) -> float:
    arr = delta.isel(time=slice(0, min(sample_times, delta.sizes["time"]))).values.astype("float64")
    pairs = []
    for a, b in [(arr[:, :, :-1], arr[:, :, 1:]), (arr[:, :-1, :], arr[:, 1:, :])]:
        x = a.ravel()
        y = b.ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            pairs.append(np.corrcoef(x[mask], y[mask])[0, 1])
    return float(np.nanmean(pairs)) if pairs else np.nan


def semivariance_short(delta: xr.DataArray, sample_times: int = 96) -> tuple[float, float]:
    arr = delta.isel(time=slice(0, min(sample_times, delta.sizes["time"]))).values.astype("float64")
    dx = arr[:, :, 1:] - arr[:, :, :-1]
    dy = arr[:, 1:, :] - arr[:, :-1, :]
    lag1 = 0.5 * np.nanmean(np.concatenate([dx.ravel() ** 2, dy.ravel() ** 2]))
    dx4 = arr[:, :, 4:] - arr[:, :, :-4]
    dy4 = arr[:, 4:, :] - arr[:, :-4, :]
    lag4 = 0.5 * np.nanmean(np.concatenate([dx4.ravel() ** 2, dy4.ravel() ** 2]))
    return float(lag1), float(lag4)


def spatial_structure(label: str, resolution_km: float, pv: xr.DataArray) -> dict[str, float | str]:
    delta = pv.diff("time")
    lag1, lag4 = semivariance_short(delta)
    return {
        "product": label,
        "nominal_resolution_km": resolution_km,
        "delta_variance": float(delta.var()),
        "semivariance_lag1_grid": lag1,
        "semivariance_lag4_grid": lag4,
        "nearest_neighbor_corr": nearest_neighbor_corr(delta),
    }


def build_coarsening_and_threshold_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    terrain = pd.read_csv(ROOT / "data/processed/terrain/terrain_classes.csv")
    base = open_pv(ROOT / "data/processed/pv_grid/2025-07/pv_pu.nc")
    era5 = open_pv(ROOT / "data/processed/era5_on_tianji_pv_grid/2025-07/pv_pu.nc")
    products = [
        RampProduct("Tianji_2.5km", base, 2.5),
        RampProduct("Tianji_coarsened_5km", coarsen_and_regrid(base, 2), 5.0),
        RampProduct("Tianji_coarsened_10km", coarsen_and_regrid(base, 4), 10.0),
        RampProduct("Tianji_coarsened_25km", coarsen_and_regrid(base, 10), 25.0),
        RampProduct("Tianji_coarsened_0.25deg_like", coarsen_and_regrid(base, 11), 27.5),
        RampProduct("ERA5_regridded_2.5km", era5, 27.5),
    ]
    summary_rows = []
    terrain_frames = []
    threshold_rows = []
    structure_rows = []
    for product in products:
        structure_rows.append(spatial_structure(product.label, product.nominal_resolution_km, product.pv))
        for threshold in [0.2, 0.3, 0.4]:
            events = detect_events(product.pv, threshold)
            clustered, clusters = cluster_events(events)
            threshold_rows.append(product_summary(product.label, product.nominal_resolution_km, product.pv, events, clusters, threshold))
            if np.isclose(threshold, 0.3):
                summary_rows.append(product_summary(product.label, product.nominal_resolution_km, product.pv, events, clusters, threshold))
                terrain_frames.append(terrain_summary(product.label, product.nominal_resolution_km, events, terrain, threshold))
        print(f"[OK] processed {product.label}")
    summary = pd.DataFrame(summary_rows)
    terrain_summary_df = pd.concat(terrain_frames, ignore_index=True)
    threshold = pd.DataFrame(threshold_rows)
    structure = pd.DataFrame(structure_rows)
    base_lag1 = structure.loc[structure["product"].eq("Tianji_2.5km"), "semivariance_lag1_grid"].iloc[0]
    structure["short_scale_variance_retention_vs_tianji"] = structure["semivariance_lag1_grid"] / base_lag1
    return summary, terrain_summary_df, threshold, structure


def build_monthly_table() -> pd.DataFrame:
    rows = []
    terrain = pd.read_csv(ROOT / "data/processed/terrain/terrain_classes.csv")
    for base_month in ["2025-04", "2025-07", "2025-10"]:
        month = base_month
        pv_path = ROOT / f"data/processed/pv_grid/{month}/pv_pu.nc"
        status = "complete"
        if not pv_path.exists():
            month = f"{base_month}_partial"
            pv_path = ROOT / f"data/processed/pv_grid/{month}/pv_pu.nc"
            status = "partial"
        if not pv_path.exists():
            continue
        pv = open_pv(pv_path)
        events = detect_events(pv, 0.3)
        _, clusters = cluster_events(events)
        row = product_summary(f"Tianji_{month}", 2.5, pv, events, clusters, 0.3)
        row["month"] = month
        row["status"] = status
        rows.append(row)
        terrain_summary(f"Tianji_{month}", 2.5, events, terrain, 0.3).to_csv(OUT / f"monthly_terrain_{month}.csv", index=False)
        print(f"[OK] monthly {month}")
    cols = ["month", "status", "time_steps", "grid_cells", "event_count", "event_rate_per_grid_hour", "cluster_count", "median_cluster_area_km2", "p95_simultaneity", "up_share"]
    return pd.DataFrame(rows)[cols]


def copy_supporting_tables() -> None:
    mappings = {
        "pv_model_sensitivity.csv": ROOT / "data/analysis/pv_model_compare/pv_model_ramp_summary.csv",
        "stdbscan_sensitivity.csv": ROOT / "data/analysis/stdbscan_sensitivity/stdbscan_parameter_scan.csv",
        "nasa_power_model_metrics.csv": ROOT / "data/analysis/nasa_power_benchmark/nasa_power_model_metrics.csv",
        "nasa_power_ramp_metrics.csv": ROOT / "data/analysis/nasa_power_benchmark/nasa_power_ghi_ramp_metrics.csv",
        "ieee33_application_delta.csv": ROOT / "data/analysis/ieee33_application_delta/hosting_delta_summary.csv",
        "ieee33_paired_scenarios.csv": ROOT / "data/analysis/ieee33_application_delta/paired_hosting_delta.csv",
    }
    for out_name, src in mappings.items():
        if src.exists():
            pd.read_csv(src).to_csv(OUT / out_name, index=False)
            print(f"[OK] copied {src} -> {OUT / out_name}")


def plot_core(summary: pd.DataFrame, threshold: pd.DataFrame, structure: pd.DataFrame, monthly: pd.DataFrame) -> None:
    coarsened = summary[summary["product"].str.startswith("Tianji")].copy()
    plt.figure(figsize=(8.2, 4.8))
    plt.plot(coarsened["nominal_resolution_km"], coarsened["event_count"], marker="o", label="Ramp events")
    plt.xlabel("Nominal spatial resolution (km)")
    plt.ylabel("Ramp event count")
    plt.title("Controlled Tianji spatial coarsening")
    plt.grid(alpha=0.25)
    savefig(FIG / "fig2a_tianji_coarsening_event_count.png")

    plt.figure(figsize=(8.2, 4.8))
    plt.plot(coarsened["nominal_resolution_km"], coarsened["cluster_count"], marker="o", label="Clusters", color="#f58518")
    plt.xlabel("Nominal spatial resolution (km)")
    plt.ylabel("Cluster count")
    plt.title("Cluster merging under spatial coarsening")
    plt.grid(alpha=0.25)
    savefig(FIG / "fig2b_tianji_coarsening_cluster_count.png")

    plt.figure(figsize=(8.4, 4.8))
    for product, group in threshold.groupby("product"):
        if product in ["Tianji_2.5km", "Tianji_coarsened_10km", "Tianji_coarsened_25km", "ERA5_regridded_2.5km"]:
            plt.plot(group["threshold"], group["event_count"], marker="o", label=product)
    plt.xlabel("Ramp threshold (p.u./h)")
    plt.ylabel("Event count")
    plt.title("Ramp threshold sensitivity")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.25)
    savefig(FIG / "fig5a_threshold_sensitivity.png")

    plt.figure(figsize=(8.2, 4.8))
    plt.bar(structure["product"], structure["short_scale_variance_retention_vs_tianji"], color="#4c78a8")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Lag-1 semivariance / Tianji")
    plt.title("Short-scale spatial variance retention")
    savefig(FIG / "fig5b_spatial_structure.png")

    if not monthly.empty:
        plt.figure(figsize=(7.8, 4.6))
        plt.bar(monthly["month"], monthly["event_count"], color="#54a24b")
        plt.ylabel("Ramp event count")
        plt.title("Multi-month Tianji diagnostics")
        savefig(FIG / "figS_monthly_diagnostics.png")


def main() -> None:
    ensure_dirs()
    summary, terrain_df, threshold, structure = build_coarsening_and_threshold_tables()
    monthly = build_monthly_table()
    summary.to_csv(OUT / "controlled_tianji_coarsening_summary.csv", index=False)
    terrain_df.to_csv(OUT / "controlled_tianji_coarsening_terrain_summary.csv", index=False)
    threshold.to_csv(OUT / "threshold_sensitivity_summary.csv", index=False)
    structure.to_csv(OUT / "spatial_structure_summary.csv", index=False)
    monthly.to_csv(OUT / "monthly_diagnostics_summary.csv", index=False)
    copy_supporting_tables()
    plot_core(summary, threshold, structure, monthly)
    print(f"[DONE] DPV ramp resolution-sensitivity experiment outputs in {OUT} and {FIG}")


if __name__ == "__main__":
    main()
