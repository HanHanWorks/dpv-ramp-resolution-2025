#!/usr/bin/env python3
"""Recompute the ERA5-vs-Tianji footprint comparison on a common hour mask."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


from project_paths import find_project_root
ROOT = find_project_root()
HERE = Path(__file__).resolve().parent
STATIONS = ROOT / "paper_writing_package_seta_major_revision/01_data/station_reference/integrated_15site_validation_hourly_repaired.csv"
MONTHS = ["2025-04", "2025-07"]
THRESHOLD = 200.0
TOLERANCE_H = 1
BOOTSTRAPS = 50000
SEED = 261113
PRODUCTS = {
    "Tianji_2.5km": ("tianji_monthly", 1, "native"),
    "Tianji_5km": ("tianji_monthly", 2, "native"),
    "Tianji_10km": ("tianji_monthly", 4, "native"),
    "Tianji_25km": ("tianji_monthly", 10, "native"),
    "Tianji_27.5km": ("tianji_monthly", 11, "native"),
    "ERA5_native_27.5km": ("era5_monthly", 1, "native"),
    "ERA5_regridded_2.5km": ("era5_on_tianji_monthly", 1, "nearest"),
}
MONTH_DIRS = {
    "2025-04": {
        "tianji_monthly": "2025-04",
        "era5_monthly": "2025-04",
        "era5_on_tianji_monthly": "2025-04_partial",
    },
    "2025-07": {
        "tianji_monthly": "2025-07",
        "era5_monthly": "2025-07",
        "era5_on_tianji_monthly": "2025-07",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_netcdf5(path):
    with h5py.File(str(path), "r") as source:
        units = source["time"].attrs["units"]
        if isinstance(units, bytes):
            units = units.decode("utf-8")
        origin = pd.Timestamp(units.split("hours since ", 1)[1])
        times = pd.DatetimeIndex(origin + pd.to_timedelta(source["time"][:], unit="h"))
        times += pd.Timedelta(hours=8)
        return (
            times,
            np.asarray(source["grid_xt"][:], dtype=float),
            np.asarray(source["grid_yt"][:], dtype=float),
            np.asarray(source["DSWRFsfc"][:], dtype=float),
        )


def nearest_index(coords, value):
    return int(np.abs(coords - value).argmin())


def sample_product(values, lon_values, lat_values, lon, lat, factor, method):
    ix = nearest_index(lon_values, lon)
    iy = nearest_index(lat_values, lat)
    if method == "nearest" or factor == 1:
        return values[:, iy, ix]
    y0 = (iy // factor) * factor
    x0 = (ix // factor) * factor
    return values[:, y0 : min(y0 + factor, values.shape[1]), x0 : min(x0 + factor, values.shape[2])].mean(axis=(1, 2))


def sample_centered_footprint(values, lon_values, lat_values, lon, lat, size):
    """Match existing remote helper: centered odd n x n window mean."""
    ix = nearest_index(lon_values, lon)
    iy = nearest_index(lat_values, lat)
    radius = size // 2
    x0, x1 = max(0, ix - radius), min(len(lon_values), ix + radius + 1)
    y0, y1 = max(0, iy - radius), min(len(lat_values), iy + radius + 1)
    return values[:, y0:y1, x0:x1].mean(axis=(1, 2))


def detect_events(values, index, pair_mask):
    events = []
    deltas = np.diff(values)
    for i in range(1, len(index)):
        if pair_mask[i] and np.isfinite(deltas[i - 1]) and abs(deltas[i - 1]) >= THRESHOLD:
            events.append(index[i])
    return events


def match(observed, model):
    tolerance = pd.Timedelta(hours=TOLERANCE_H)
    remaining = list(sorted(model))
    hits = 0
    for observed_time in sorted(observed):
        candidates = [t for t in remaining if abs(t - observed_time) <= tolerance]
        if candidates:
            chosen = min(candidates, key=lambda t: abs(t - observed_time))
            remaining.remove(chosen)
            hits += 1
    misses = len(observed) - hits
    false_alarms = len(remaining)
    denominator = hits + misses + false_alarms
    return {
        "observed_events": len(observed),
        "modelled_events": hits + false_alarms,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "csi": hits / float(denominator) if denominator else np.nan,
        "pod": hits / float(hits + misses) if hits + misses else np.nan,
        "far": false_alarms / float(hits + false_alarms) if hits + false_alarms else np.nan,
    }


def paired_bootstrap(frame, left, right, rng):
    """CI of right-left CSI, sampling station clusters and retaining months."""
    pivot = frame.pivot_table(
        index=["station_id", "month"], columns="series", values="csi", aggfunc="first"
    )
    stations = sorted(pivot.index.get_level_values("station_id").unique())
    months = sorted(pivot.index.get_level_values("month").unique())
    pivot = pivot.reindex(pd.MultiIndex.from_product([stations, months], names=["station_id", "month"]))
    if pivot[[left, right]].isna().any().any():
        raise RuntimeError("Missing paired station-month CSI for {} versus {}".format(right, left))
    diff = (pivot[right] - pivot[left]).to_numpy().reshape(len(stations), len(months))
    point = float(diff.mean())
    boot = np.empty(BOOTSTRAPS, dtype=float)
    for start in range(0, BOOTSTRAPS, 500):
        n = min(500, BOOTSTRAPS - start)
        draw = rng.integers(0, len(stations), size=(n, len(stations)))
        boot[start : start + n] = diff[draw].mean(axis=(1, 2))
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), len(stations), len(months)


def main():
    station = pd.read_csv(STATIONS, low_memory=False)
    station = station[station["common_period_flag"].eq(1) & station["usable_absolute_event_pairing"].eq(1)].copy()
    station["local_time"] = pd.to_datetime(
        station["datetime_hubei_local_fixed"].astype(str), utc=True
    ).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    # Set complete files are loaded once per month; all products share the same
    # station-level common paired-hour mask at shift=0.
    rows = []
    masks = []
    inputs = {str(STATIONS.relative_to(ROOT)): sha256(STATIONS)}

    for month in MONTHS:
        month_station = station[station["local_time"].dt.strftime("%Y-%m").eq(month)]
        cache = {}
        for product, (folder, _, _) in PRODUCTS.items():
            relative = MONTH_DIRS[month][folder]
            path = ROOT / "data/processed" / folder / relative / "DSWRFsfc.nc"
            inputs[str(path.relative_to(ROOT))] = sha256(path)
            cache[product] = (path, read_netcdf5(path))

        for station_id, sg in month_station.groupby("station_id"):
            sg = sg.sort_values("local_time")
            site = str(sg["hubei_site"].iloc[0])
            qc = str(sg["solar_qc_group"].iloc[0])
            lon = float(sg["hubei_lon"].iloc[0])
            lat = float(sg["hubei_lat"].iloc[0])
            obs = pd.Series(sg["ghi_wm2"].to_numpy(float), index=pd.DatetimeIndex(sg["local_time"]))

            sampled = {}
            for product, (folder, factor, method) in PRODUCTS.items():
                times, lon_values, lat_values, values = cache[product][1]
                vector = sample_product(values, lon_values, lat_values, lon, lat, factor, method)
                sampled[product] = pd.Series(vector, index=times)

            tianji_times, tianji_lons, tianji_lats, tianji_values = cache["Tianji_2.5km"][1]
            for footprint in [3, 5]:
                vector = sample_centered_footprint(tianji_values, tianji_lons, tianji_lats, lon, lat, footprint)
                sampled["Tianji_2.5km_{}x{}".format(footprint, footprint)] = pd.Series(vector, index=tianji_times)

            month_start = pd.Timestamp(month + "-01 00:00:00")
            month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
            # All hourly times from fields and station data within target month.
            start = month_start
            end = month_end
            index = pd.date_range(start, end, freq="h")
            obs_hourly = obs.reindex(index)
            aligned = {name: values.reindex(index) for name, values in sampled.items()}
            common = obs_hourly.notna()
            for values in aligned.values():
                common &= values.notna()
            common_hours = common.to_numpy(bool)
            pair_mask = np.zeros(len(index), dtype=bool)
            pair_mask[1:] = common_hours[1:] & common_hours[:-1]
            # Hour zero of the month cannot be paired with the previous month's
            # record; this mask is constructed within a month, so it stays false.
            masks.append(
                {
                    "month": month,
                    "station_id": int(station_id),
                    "site": site,
                    "qc_group": qc,
                    "common_hours_all_products_and_footprints": int(common_hours.sum()),
                    "valid_consecutive_same_month_pairs": int(pair_mask.sum()),
                    "first_valid_common_hour": index[common_hours][0].isoformat() if common_hours.any() else "",
                    "last_valid_common_hour": index[common_hours][-1].isoformat() if common_hours.any() else "",
                }
            )
            if not pair_mask.any():
                continue
            observed_events = detect_events(obs_hourly.to_numpy(float), index, pair_mask)
            for series, values in aligned.items():
                modeled_events = detect_events(values.to_numpy(float), index, pair_mask)
                scores = match(observed_events, modeled_events)
                rows.append(
                    {
                        "month": month,
                        "station_id": int(station_id),
                        "site": site,
                        "solar_qc_group": qc,
                        "series": series,
                        "threshold_wm2h": THRESHOLD,
                        "tolerance_h": TOLERANCE_H,
                        "shift_h": 0,
                        "common_valid_hours": int(common_hours.sum()),
                        "common_consecutive_hour_pairs": int(pair_mask.sum()),
                        **scores,
                    }
                )

    detail = pd.DataFrame(rows)
    detail_path = HERE / "footprint_era5_common_mask_station_month.csv"
    detail.to_csv(detail_path, index=False)
    mask_path = HERE / "footprint_era5_common_mask_hour_counts.csv"
    pd.DataFrame(masks).to_csv(mask_path, index=False)

    selected = [
        "Tianji_2.5km",
        "Tianji_2.5km_3x3",
        "Tianji_2.5km_5x5",
        "ERA5_native_27.5km",
        "ERA5_regridded_2.5km",
        "Tianji_27.5km",
    ]
    summaries = []
    rng = np.random.default_rng(SEED)
    # Paired station-month contrasts. Keep the
    # primary and secondary QC groups separate, as in the station benchmark.
    contrasts = []
    for footprint in ["Tianji_2.5km", "Tianji_2.5km_3x3", "Tianji_2.5km_5x5"]:
        for comparator in ["ERA5_native_27.5km", "ERA5_regridded_2.5km"]:
            contrasts.append((footprint, comparator))
    for footprint in ["Tianji_2.5km_3x3", "Tianji_2.5km_5x5"]:
        contrasts.append((footprint, "Tianji_27.5km"))
    for qc_group, qc_data in detail.groupby("solar_qc_group"):
        # Descriptive product results within QC group.
        for series, g in qc_data[qc_data["series"].isin(selected)].groupby("series"):
            summaries.append(
                {
                    "row_type": "product",
                    "solar_qc_group": qc_group,
                    "series": series,
                    "comparison": "",
                    "stations": g["station_id"].nunique(),
                    "station_months": len(g),
                    "macro_station_month_csi": g["csi"].mean(),
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "paired_delta_csi": np.nan,
                    "paired_delta_ci95_low": np.nan,
                    "paired_delta_ci95_high": np.nan,
                    "observed_events_sum": int(g["observed_events"].sum()),
                    "modelled_events_sum": int(g["modelled_events"].sum()),
                }
            )

        for footprint, comparator in contrasts:
            left = qc_data[qc_data["series"].eq(footprint)].copy()
            right = qc_data[qc_data["series"].eq(comparator)].copy()
            paired = left.merge(
                right,
                on=["month", "station_id", "site", "solar_qc_group", "threshold_wm2h", "tolerance_h", "shift_h", "common_valid_hours", "common_consecutive_hour_pairs"],
                suffixes=("_left", "_right"),
            )
            # Bootstrap on paired difference directly, station cluster preserving months.
            wide = paired.pivot_table(index=["station_id", "month"], values=["csi_left", "csi_right"])
            stations = sorted(wide.index.get_level_values("station_id").unique())
            months = sorted(wide.index.get_level_values("month").unique())
            wide = wide.reindex(pd.MultiIndex.from_product([stations, months], names=["station_id", "month"]))
            delta = (wide["csi_right"] - wide["csi_left"]).to_numpy().reshape(len(stations), len(months))
            boot = np.empty(BOOTSTRAPS, dtype=float)
            for start in range(0, BOOTSTRAPS, 500):
                n = min(500, BOOTSTRAPS - start)
                draw = rng.integers(0, len(stations), size=(n, len(stations)))
                boot[start : start + n] = delta[draw].mean(axis=(1, 2))
            summaries.append(
                {
                    "row_type": "paired_contrast",
                    "solar_qc_group": qc_group,
                    "series": comparator,
                    "comparison": "{} minus {}".format(comparator, footprint),
                    "stations": len(stations),
                    "station_months": len(paired),
                    "macro_station_month_csi": paired["csi_right"].mean(),
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "paired_delta_csi": delta.mean(),
                    "paired_delta_ci95_low": float(np.quantile(boot, 0.025)),
                    "paired_delta_ci95_high": float(np.quantile(boot, 0.975)),
                    "observed_events_sum": int(paired["observed_events_left"].sum()),
                    "modelled_events_sum": int(paired["modelled_events_right"].sum()),
                }
            )

    summary = pd.DataFrame(summaries)
    summary_path = HERE / "footprint_era5_common_mask_summary.csv"
    summary.to_csv(summary_path, index=False)
    metadata = {
        "analysis": "shift=0 station-hourly ramp re-detection comparing Tianji 2.5 km centered 1x1/3x3/5x5 footprints with ERA5 products",
        "inputs_sha256": inputs,
        "selection": "April and July 2025; 7 primary-QC stations; 200 W m-2 h-1 ramp threshold; 1 h event matching tolerance",
        "common_mask": "station GHI plus all seven canonical product series and the 3x3/5x5 Tianji footprint series must be valid at both adjacent hours; all pairs are within the same month; no interpolation and no differencing across missing-hour gaps",
        "footprint_definition": "odd centered n x n window around nearest Tianji 2.5 km grid cell; arithmetic cell mean; identical to scripts/run_remote_station_resolution_metrics.py sample_footprint",
        "ERA5_products": "native ERA5 grid at nearest cell; ERA5 already interpolated to Tianji grid at nearest cell, matching canonical product definitions",
        "bootstrap": "50,000 paired station-cluster percentile replicates, seed 261113; resample station IDs with replacement separately by primary/secondary QC group and retain both April and July rows",
        "positive_paired_delta": "ERA5 comparison has higher macro station-month CSI than the stated Tianji footprint",
        "caveat": "This directly addresses event-skill comparison under a shared hour mask; low station count means bootstrap inference is exploratory.",
        "outputs": [detail_path.name, mask_path.name, summary_path.name],
    }
    (HERE / "footprint_era5_common_mask_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
