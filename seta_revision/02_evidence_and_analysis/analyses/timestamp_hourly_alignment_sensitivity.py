#!/usr/bin/env python3
"""Re-detect station-referenced GHI ramps after hourly model timestamp shifts.

Uses locally retained station observations and hourly NetCDF5 fields. Ramps are
computed only when consecutive valid hours are present. Every product and
tested shift shares one station-specific timestamp mask within each month.
"""

from __future__ import print_function

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


from project_paths import find_project_root
ROOT = find_project_root()
OUT = Path(__file__).resolve().parent
STATIONS = ROOT / "paper_writing_package_seta_major_revision/01_data/station_reference/integrated_15site_validation_hourly_repaired.csv"
MONTHS = ["2025-04", "2025-07"]
SHIFTS_H = [-1, 0, 1]
THRESHOLD = 200.0
TOLERANCE_H = 1
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
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_netcdf5(path):
    """Read the small set of coordinate and field arrays through h5py."""
    with h5py.File(str(path), "r") as source:
        time_ds = source["time"]
        units = time_ds.attrs["units"]
        if isinstance(units, bytes):
            units = units.decode("utf-8")
        match = re.match(r"hours since (.+)", units)
        if not match:
            raise ValueError("Unsupported time units in {}: {}".format(path, units))
        origin = pd.Timestamp(match.group(1))
        times = origin + pd.to_timedelta(time_ds[:], unit="h")
        times = pd.DatetimeIndex(times) + pd.Timedelta(hours=8)
        lon = np.asarray(source["grid_xt"][:], dtype=float)
        lat = np.asarray(source["grid_yt"][:], dtype=float)
        field = np.asarray(source["DSWRFsfc"][:], dtype=float)
    return times, lon, lat, field


def sample_station(values, lon_values, lat_values, lon, lat, factor, method):
    ix = int(np.abs(lon_values - lon).argmin())
    iy = int(np.abs(lat_values - lat).argmin())
    if method == "nearest" or factor == 1:
        return values[:, iy, ix]
    y0 = (iy // factor) * factor
    x0 = (ix // factor) * factor
    y1 = min(y0 + factor, values.shape[1])
    x1 = min(x0 + factor, values.shape[2])
    return values[:, y0:y1, x0:x1].mean(axis=(1, 2))


def greedy_counts(observed_times, model_times):
    tolerance = pd.Timedelta(hours=TOLERANCE_H)
    remaining = list(sorted(model_times))
    hits = 0
    for observed_time in sorted(observed_times):
        candidates = [
            model_time
            for model_time in remaining
            if abs(model_time - observed_time) <= tolerance
        ]
        if candidates:
            chosen = min(candidates, key=lambda value: abs(value - observed_time))
            remaining.remove(chosen)
            hits += 1
    misses = len(observed_times) - hits
    false_alarms = len(remaining)
    denominator = hits + misses + false_alarms
    return {
        "obs_events": len(observed_times),
        "model_events": hits + false_alarms,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "pod": hits / float(hits + misses) if hits + misses else np.nan,
        "far": false_alarms / float(hits + false_alarms)
        if hits + false_alarms
        else np.nan,
        "csi": hits / float(denominator) if denominator else np.nan,
    }


def event_times(values, times, pair_mask):
    """Return threshold events whose two source hours are in common mask."""
    output = []
    for i in range(1, len(times)):
        if not pair_mask[i]:
            continue
        delta = values[i] - values[i - 1]
        if abs(delta) >= THRESHOLD:
            output.append(times[i])
    return output


def main():
    station = pd.read_csv(STATIONS, low_memory=False)
    station = station[
        station["common_period_flag"].eq(1)
        & station["usable_absolute_event_pairing"].eq(1)
    ].copy()
    station["local_time"] = pd.to_datetime(
        station["datetime_hubei_local_fixed"].astype(str), utc=True
    ).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)

    rows = []
    common_mask_rows = []
    for month in MONTHS:
        month_station = station[station["local_time"].dt.strftime("%Y-%m").eq(month)]
        product_cache = {}
        month_path_map = {}
        for product, (folder, factor, method) in PRODUCTS.items():
            relative_dir = MONTH_DIRS[month][folder]
            path = ROOT / "data/processed" / folder / relative_dir / "DSWRFsfc.nc"
            if not path.exists():
                raise FileNotFoundError(str(path))
            product_cache[product] = (path, factor, method)
            month_path_map[product] = path

        field_cache = {}
        for product, (path, factor, method) in product_cache.items():
            cache_key = str(path)
            if cache_key not in field_cache:
                field_cache[cache_key] = read_netcdf5(path)

        for station_id, station_rows in month_station.groupby("station_id"):
            station_rows = station_rows.sort_values("local_time")
            station_site = str(station_rows["hubei_site"].iloc[0])
            qc_group = str(station_rows["solar_qc_group"].iloc[0])
            station_lon = float(station_rows["hubei_lon"].iloc[0])
            station_lat = float(station_rows["hubei_lat"].iloc[0])
            obs = pd.Series(
                station_rows["ghi_wm2"].to_numpy(dtype=float),
                index=pd.DatetimeIndex(station_rows["local_time"]),
            )

            sampled = {}
            for product, (path, factor, method) in product_cache.items():
                times, lon_values, lat_values, values = field_cache[str(path)]
                sampled_values = sample_station(
                    values,
                    lon_values,
                    lat_values,
                    station_lon,
                    station_lat,
                    factor,
                    method,
                )
                # NetCDF's stored valid-time coordinate is UTC; canonical
                # analysis adds eight hours to compare with Beijing station time.
                series = pd.Series(sampled_values, index=times).sort_index()
                sampled[product] = series

            # Construct a shared local hourly support for the complete month.
            all_dates = set(obs.index)
            for product, series in sampled.items():
                all_dates.update(series.index)
            full_index = pd.date_range(
                month + "-01 00:00:00",
                (pd.Timestamp(month + "-01") + pd.offsets.MonthEnd(1)).normalize()
                + pd.Timedelta(hours=23),
                freq="h",
            )
            # Include boundary hours needed by ±1 h shifts, then restrict event
            # timestamps to target-month hour labels.
            extended_index = pd.date_range(
                full_index[0] - pd.Timedelta(hours=2),
                full_index[-1] + pd.Timedelta(hours=2),
                freq="h",
            )
            obs_full = obs.reindex(extended_index)
            aligned = {}
            for product, series in sampled.items():
                shifted_values = {}
                for shift in SHIFTS_H:
                    # A positive shift moves model timestamps later relative to
                    # station timestamps, equivalent to a delayed model clock.
                    shifted_values[shift] = series.copy()
                    shifted_values[shift].index = shifted_values[shift].index + pd.Timedelta(hours=shift)
                    shifted_values[shift] = shifted_values[shift].reindex(extended_index)
                aligned[product] = shifted_values

            # All products and all shifts must be present at t and t-1; station
            # data must also be present at both hours. Exclude interpolated gaps.
            common = obs_full.notna()
            for product in PRODUCTS:
                for shift in SHIFTS_H:
                    common &= aligned[product][shift].notna()
            common_array = common.to_numpy(dtype=bool)
            pair_mask = np.zeros(len(extended_index), dtype=bool)
            pair_mask[1:] = common_array[1:] & common_array[:-1]
            target_month = extended_index.strftime("%Y-%m") == month
            same_month_pair = np.zeros(len(extended_index), dtype=bool)
            same_month_pair[1:] = target_month[1:] & target_month[:-1]
            pair_mask &= same_month_pair
            common_index = extended_index[pair_mask]
            common_mask_rows.append(
                {
                    "month": month,
                    "station_id": int(station_id),
                    "hubei_site": station_site,
                    "solar_qc_group": qc_group,
                    "common_valid_consecutive_hour_pairs": int(pair_mask.sum()),
                    "common_first_hour": common_index.min().isoformat() if len(common_index) else "",
                    "common_last_hour": common_index.max().isoformat() if len(common_index) else "",
                }
            )
            if pair_mask.sum() == 0:
                continue
            obs_values = obs_full.to_numpy(dtype=float)
            obs_events = event_times(obs_values, extended_index, pair_mask)
            for product in PRODUCTS:
                for shift in SHIFTS_H:
                    model_values = aligned[product][shift].to_numpy(dtype=float)
                    model_events = event_times(model_values, extended_index, pair_mask)
                    # Both observed and model event sets are restricted to the
                    # month of the event label after the shift is applied.
                    month_start = pd.Timestamp(month + "-01 00:00:00")
                    month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
                    obs_in_month = [t for t in obs_events if month_start <= t <= month_end]
                    model_in_month = [t for t in model_events if month_start <= t <= month_end]
                    counts = greedy_counts(obs_in_month, model_in_month)
                    rows.append(
                        {
                            "month": month,
                            "station_id": int(station_id),
                            "hubei_site": station_site,
                            "solar_qc_group": qc_group,
                            "product": product,
                            "threshold_wm2h": THRESHOLD,
                            "tolerance_h": TOLERANCE_H,
                            "model_timestamp_shift_h": shift,
                            "common_valid_consecutive_hour_pairs": int(pair_mask.sum()),
                            **counts,
                        }
                    )

    details = pd.DataFrame(rows)
    details_path = OUT / "timestamp_hourly_alignment_sensitivity.csv"
    details.to_csv(details_path, index=False)
    mask_path = OUT / "timestamp_hourly_alignment_common_mask.csv"
    pd.DataFrame(common_mask_rows).to_csv(mask_path, index=False)

    summary_rows = []
    for qc_group, group in details.groupby("solar_qc_group"):
        for product, prod in group.groupby("product"):
            for shift, shift_group in prod.groupby("model_timestamp_shift_h"):
                station_month = shift_group.groupby(["station_id", "month"], as_index=False).agg(
                    csi=("csi", "mean"), pod=("pod", "mean"), far=("far", "mean"),
                    obs_events=("obs_events", "sum"), model_events=("model_events", "sum"),
                    hits=("hits", "sum"), misses=("misses", "sum"), false_alarms=("false_alarms", "sum"),
                )
                summary_rows.append(
                    {
                        "coverage": "April_and_July_common_station_months",
                        "solar_qc_group": qc_group,
                        "product": product,
                        "threshold_wm2h": THRESHOLD,
                        "tolerance_h": TOLERANCE_H,
                        "model_timestamp_shift_h": int(shift),
                        "stations": int(station_month["station_id"].nunique()),
                        "station_months": int(len(station_month)),
                        "macro_station_month_csi": float(station_month["csi"].mean()),
                        "macro_station_month_pod": float(station_month["pod"].mean()),
                        "macro_station_month_far": float(station_month["far"].mean()),
                        "obs_events_sum_descriptive": int(station_month["obs_events"].sum()),
                        "model_events_sum_descriptive": int(station_month["model_events"].sum()),
                        "hits_sum_descriptive": int(station_month["hits"].sum()),
                        "misses_sum_descriptive": int(station_month["misses"].sum()),
                        "false_alarms_sum_descriptive": int(station_month["false_alarms"].sum()),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary_path = OUT / "timestamp_hourly_alignment_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    inputs = {str(STATIONS.relative_to(ROOT)): sha256(STATIONS)}
    for product, (folder, _, _) in PRODUCTS.items():
        for month in MONTHS:
            path = ROOT / "data/processed" / folder / MONTH_DIRS[month][folder] / "DSWRFsfc.nc"
            inputs[str(path.relative_to(ROOT))] = sha256(path)
    metadata = {
        "analysis_type": "hourly timestamp-shift re-detection using locally retained station and gridded data",
        "inputs_sha256": inputs,
        "selection": {
            "months": MONTHS,
            "station_flags": "common_period_flag=1 and usable_absolute_event_pairing=1",
            "products": list(PRODUCTS),
            "threshold_wm2h": THRESHOLD,
            "matching_tolerance_h": TOLERANCE_H,
            "timestamp_shifts_h": SHIFTS_H,
        },
        "time_alignment": "netCDF UTC valid-time coordinates shifted by +8 h to Beijing local time, then tested relative model shifts of -1, 0, +1 h",
        "sampling": "nearest grid cell for native/Tianji products; Tianji coarsened by contiguous 2x2, 4x4, 10x10, 11x11 blocks; ERA5-on-Tianji grid sampled at nearest cell",
        "common_mask": "within each station-month, require station and every product at every tested shift to be valid at both t and t-1; both hours must be in the target month, excluding month-boundary differences",
        "event_definition": "absolute one-hour GHI difference >= 200 W m-2 h-1; only adjacent valid hourly pairs",
        "matcher": "chronological greedy nearest unmatched event, matching scripts/run_seta_canonical_station_scan.py",
        "aggregation": "macro mean across station-month CSI/POD/FAR; primary and secondary QC groups reported separately",
        "limitations": [
            "This is a re-detection sensitivity at the hourly timestamp level; station QC and timestamp interpretation remain source assumptions.",
            "ERA5 is available only for April and July in the local archive, so comparisons use these common months.",
            "The common mask is intentionally stricter than product-specific canonical masks and may alter absolute scores.",
            "No plant-level AC power observations are used; the validation target is station GHI ramps.",
        ],
        "outputs": [details_path.name, summary_path.name, mask_path.name],
    }
    metadata_path = OUT / "timestamp_hourly_alignment_sensitivity_metadata.json"
    with open(str(metadata_path), "w") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print("common mask rows:", len(common_mask_rows))
    print("event metric rows:", len(details))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
