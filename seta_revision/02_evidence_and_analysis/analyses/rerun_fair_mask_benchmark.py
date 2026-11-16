#!/usr/bin/env python3
"""Re-detect station-referenced GHI ramps under alternative time labels.

This portable release copy re-pairs hourly station observations to the saved
gridded products and detects events at the adopted zero-shift convention only. It uses the Hubei station file retained in the
major-revision data package and reads NetCDF/HDF5 arrays directly with h5py.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


from project_paths import find_project_root
ROOT = find_project_root()
OUT = Path(__file__).resolve().parent
STATIONS = ROOT / "paper_writing_package_seta_major_revision/01_data/station_reference/integrated_15site_validation_hourly_repaired.csv"
DATA = ROOT / "data"
MONTHS = ["2025-04", "2025-07"]
SHIFTS = [0]  # Adopted station convention only; clock sensitivity is a separate script.
THRESHOLD = 200.0
TOLERANCES = [1, 2]
PRODUCTS = {
    "Tianji_2.5km": ("tianji_monthly", 1, "native"),
    "Tianji_5km": ("tianji_monthly", 2, "block"),
    "Tianji_10km": ("tianji_monthly", 4, "block"),
    "Tianji_25km": ("tianji_monthly", 10, "block"),
    "Tianji_27.5km": ("tianji_monthly", 11, "block"),
    "ERA5_native_27.5km": ("era5_monthly", 1, "native"),
    "ERA5_regridded_2.5km": ("era5_on_tianji_monthly", 1, "native"),
}
MONTH_FOLDERS = {
    "tianji_monthly": {"2025-04": "2025-04", "2025-07": "2025-07"},
    "era5_monthly": {"2025-04": "2025-04_partial", "2025-07": "2025-07"},
    "era5_on_tianji_monthly": {"2025-04": "2025-04_partial", "2025-07": "2025-07"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_field(product: str, month: str) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    source, factor, sampling = PRODUCTS[product]
    folder = MONTH_FOLDERS[source][month]
    path = DATA / "processed" / source / folder / "DSWRFsfc.nc"
    with h5py.File(path, "r") as f:
        values = f["DSWRFsfc"][:].astype(np.float32)
        lon = f["grid_xt"][:].astype(float)
        lat = f["grid_yt"][:].astype(float)
        raw_time = f["time"][:].astype(float)
        units = f["time"].attrs["units"]
        if isinstance(units, bytes):
            units = units.decode("ascii")
    origin = units.replace("hours since ", "").replace("T", " ").replace("Z", "")
    model_times = pd.DatetimeIndex(pd.Timestamp(origin) + pd.to_timedelta(raw_time, unit="h") + pd.Timedelta(hours=8))
    return values, model_times, lon, lat


def sampled_series(product: str, field: tuple, station: pd.Series) -> pd.Series:
    values, times, lon_values, lat_values = field
    lon, lat = float(station.hubei_lon), float(station.hubei_lat)
    ix = int(np.abs(lon_values - lon).argmin())
    iy = int(np.abs(lat_values - lat).argmin())
    if PRODUCTS[product][2] == "block":
        factor = PRODUCTS[product][1]
        y0, x0 = (iy // factor) * factor, (ix // factor) * factor
        patch = values[:, y0:min(y0 + factor, values.shape[1]), x0:min(x0 + factor, values.shape[2])]
        vals = np.nanmean(patch, axis=(1, 2))
    else:
        vals = values[:, iy, ix]
    return pd.Series(vals.astype(float), index=times, name="model")


def detect(series: pd.Series) -> tuple[list[pd.Timestamp], list[float]]:
    delta = series.diff()
    # A missing observation must not turn a multi-hour change into an hourly
    # ramp. Keep only differences between adjacent timestamps exactly 1 h apart.
    adjacent_hour = series.index.to_series().diff().eq(pd.Timedelta(hours=1))
    keep = delta.notna() & adjacent_hour.to_numpy() & delta.abs().ge(THRESHOLD)
    return list(series.index[keep]), list(delta.loc[keep].astype(float))


def match(obs: list[pd.Timestamp], model: list[pd.Timestamp], tol_h: int) -> tuple[int, int, int]:
    remaining = list(sorted(model))
    hits = 0
    tol = pd.Timedelta(hours=tol_h)
    for t in sorted(obs):
        candidates = [m for m in remaining if abs(m - t) <= tol]
        if candidates:
            chosen = min(candidates, key=lambda m: abs(m - t))
            remaining.remove(chosen)
            hits += 1
    return hits, len(obs) - hits, len(remaining)


def main() -> None:
    stations = pd.read_csv(STATIONS, low_memory=False)
    stations["local_time"] = pd.to_datetime(stations["datetime_hubei_local_fixed"], utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    stations = stations[(stations["common_period_flag"] == 1) & (stations["usable_absolute_event_pairing"] == 1)]
    stations = stations[stations["month"] .isin(MONTHS)] if "month" in stations else stations
    rows: list[dict] = []
    field_cache: dict[tuple[str, str], tuple] = {}
    for month in MONTHS:
        sm = stations[stations.local_time.dt.strftime("%Y-%m").eq(month)]
        for sid, group in sm.groupby("station_id", sort=True):
            group = group.sort_values("local_time")
            site = group.iloc[0]
            observed_base = pd.Series(group.ghi_wm2.to_numpy(float), index=pd.DatetimeIndex(group.local_time))
            model_series: dict[str, pd.Series] = {}
            for product in PRODUCTS:
                key = (product, month)
                if key not in field_cache:
                    field_cache[key] = read_field(product, month)
                model_series[product] = sampled_series(product, field_cache[key], site).rename(product)
            for shift in SHIFTS:
                observed = observed_base.copy()
                observed.index = observed.index + pd.Timedelta(hours=shift)
                observed = observed.rename("obs")
                # Alongside the product-specific valid-hour comparison, retain
                # a strict all-product intersection so pairwise skill changes
                # are not caused by different hourly masks.
                product_specific = {
                    product: pd.concat([observed, model], axis=1).dropna()
                    for product, model in model_series.items()
                }
                common = pd.concat([observed, *model_series.values()], axis=1).dropna()
                scopes = {
                                        "all_products_common_hours": {
                        product: common[["obs", product]].rename(columns={product: "model"})
                        for product in PRODUCTS
                    },
                }
                for mask_scope, paired_by_product in scopes.items():
                    for product, paired in paired_by_product.items():
                        if mask_scope == "product_specific":
                            paired = paired.rename(columns={product: "model"})
                        obs_times, _ = detect(paired.obs)
                        model_times, _ = detect(paired.model)
                        for tol in TOLERANCES:
                            hits, misses, false_alarms = match(obs_times, model_times, tol)
                            denom = hits + misses + false_alarms
                            rows.append({
                                "month": month,
                                "station_id": int(sid),
                                "hubei_site": site.hubei_site,
                                "solar_qc_group": site.solar_qc_group,
                                "product": product,
                                "mask_scope": mask_scope,
                                "threshold_wm2h": THRESHOLD,
                                "tolerance_h": tol,
                                "station_time_shift_h": shift,
                                "paired_hours": len(paired),
                                "obs_events": len(obs_times),
                                "model_events": len(model_times),
                                "hits": hits,
                                "misses": misses,
                                "false_alarms": false_alarms,
                                "pod": hits / (hits + misses) if hits + misses else np.nan,
                                "far": false_alarms / (hits + false_alarms) if hits + false_alarms else np.nan,
                                "csi": hits / denom if denom else np.nan,
                            })
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "timestamp_redetection_sensitivity.csv", index=False)
    common = detail[detail.solar_qc_group.eq("primary_solar_qc")]
    summary = (common.groupby(["mask_scope", "product", "tolerance_h", "station_time_shift_h"], as_index=False)
               .agg(stations=("station_id", "nunique"), station_months=("station_id", "size"),
                    macro_csi=("csi", "mean"), macro_pod=("pod", "mean"), macro_far=("far", "mean"),
                    paired_hours=("paired_hours", "sum"), obs_events=("obs_events", "sum"),
                    model_events=("model_events", "sum"), hits=("hits", "sum"), misses=("misses", "sum"),
                    false_alarms=("false_alarms", "sum")))
    summary.to_csv(OUT / "timestamp_redetection_sensitivity_summary.csv", index=False)
    meta = {
        "analysis_type": "hourly station/model re-pairing and ramp re-detection under alternative station timestamp offsets",
        "station_input": str(STATIONS.relative_to(ROOT)),
        "station_input_sha256": sha256(STATIONS),
        "month_coverage": MONTHS,
        "qc_scope_in_summary": "seven primary-QC stations; secondary group retained in detail output only",
        "threshold_wm2h": THRESHOLD,
        "tolerances_h": TOLERANCES,
        "station_time_shifts_h": SHIFTS,
        "shift_definition": "positive shifts station observation timestamps later relative to model local timestamps",
        "products": list(PRODUCTS),
        "limitations": [
            "Offsets test timestamp-label alternatives; they cannot resolve the provider's undocumented clock/interval convention.",

            "April uses the partial-month ERA5 archive for both ERA5 variants, matching the formal common-month benchmark; this release copy uses only the explicitly reported all-product common-hour mask.",
            "A ramp is counted only when adjacent paired timestamps are exactly one hour apart, preventing differences across missing-hour gaps from being treated as hourly events.",
            "Station-cluster confidence intervals are not calculated in this diagnostic; monthly paired outputs are descriptive sensitivity estimates.",
        ],
        "model_inputs_sha256": {f"{p}/{m}": sha256(DATA / "processed" / PRODUCTS[p][0] / MONTH_FOLDERS[PRODUCTS[p][0]][m] / "DSWRFsfc.nc") for p in PRODUCTS for m in MONTHS},
    }
    (OUT / "timestamp_redetection_sensitivity_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary[summary.station_time_shift_h.isin([-1, 0, 1])].to_string(index=False))


if __name__ == "__main__":
    main()
