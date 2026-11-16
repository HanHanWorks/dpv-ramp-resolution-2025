#!/usr/bin/env python3
"""Build one canonical station-referenced ramp-skill table.

This script fixes the following analysis choices in one place:

* repaired station timestamps and QC groups;
* 100-400 W m-2 h-1 ramp thresholds;
* 0-3 h event matching tolerances;
* Tianji controlled coarsening and independent ERA5 products;
* station-month metrics, event-level detections and false alarms.

All downstream statistics and figures must use these outputs rather than
mixing results from experiments with different matching conventions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


RESOLUTIONS = {
    "2.5_km": 1,
    "5_km": 2,
    "10_km": 4,
    "25_km": 10,
    "27.5_km": 11,
}

PRODUCT_SPECS = {
    "Tianji_2.5km": {
        "source": "tianji",
        "factor": 1,
        "nominal_resolution_km": 2.5,
        "physics_family": "Tianji",
        "support_km": 2.5,
    },
    "Tianji_5km": {
        "source": "tianji",
        "factor": 2,
        "nominal_resolution_km": 5.0,
        "physics_family": "Tianji",
        "support_km": 5.0,
    },
    "Tianji_10km": {
        "source": "tianji",
        "factor": 4,
        "nominal_resolution_km": 10.0,
        "physics_family": "Tianji",
        "support_km": 10.0,
    },
    "Tianji_25km": {
        "source": "tianji",
        "factor": 10,
        "nominal_resolution_km": 25.0,
        "physics_family": "Tianji",
        "support_km": 25.0,
    },
    "Tianji_27.5km": {
        "source": "tianji",
        "factor": 11,
        "nominal_resolution_km": 27.5,
        "physics_family": "Tianji",
        "support_km": 27.5,
    },
    "ERA5_native_27.5km": {
        "source": "era5_native",
        "factor": 1,
        "nominal_resolution_km": 27.5,
        "physics_family": "ERA5",
        "support_km": 27.5,
    },
    "ERA5_regridded_2.5km": {
        "source": "era5_regridded",
        "factor": 1,
        "nominal_resolution_km": 2.5,
        "physics_family": "ERA5",
        "support_km": 27.5,
    },
}

ERA5_FOLDERS = {
    "2025-04": "2025-04_partial",
    "2025-07": "2025-07",
}


def parse_local_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype(str), errors="raise", utc=True).dt.tz_convert(
        "Asia/Shanghai"
    )


def sample_footprint(
    field: xr.DataArray,
    lon: float,
    lat: float,
) -> np.ndarray:
    lon_values = field["grid_xt"].values
    lat_values = field["grid_yt"].values
    ix = int(np.abs(lon_values - lon).argmin())
    iy = int(np.abs(lat_values - lat).argmin())
    return field.isel(grid_xt=ix, grid_yt=iy).values.astype(float)


def sample_block_mean(
    values: np.ndarray,
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    lon: float,
    lat: float,
    factor: int,
) -> np.ndarray:
    ix = int(np.abs(lon_values - lon).argmin())
    iy = int(np.abs(lat_values - lat).argmin())
    y0 = (iy // factor) * factor
    x0 = (ix // factor) * factor
    y1 = min(y0 + factor, values.shape[1])
    x1 = min(x0 + factor, values.shape[2])
    return values[:, y0:y1, x0:x1].mean(axis=(1, 2)).astype(float)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def match_events_detailed(
    observed_times: list[pd.Timestamp],
    modeled_times: list[pd.Timestamp],
    tolerance_h: int,
) -> tuple[dict[str, int], pd.DataFrame, list[pd.Timestamp]]:
    tolerance = pd.Timedelta(hours=tolerance_h)
    unmatched_model = list(modeled_times)
    rows = []
    for observed in observed_times:
        candidates = [
            modeled
            for modeled in unmatched_model
            if abs((modeled - observed).total_seconds()) <= tolerance.total_seconds()
        ]
        if candidates:
            chosen = min(
                candidates,
                key=lambda value: abs((value - observed).total_seconds()),
            )
            unmatched_model.remove(chosen)
            rows.append(
                {
                    "observed_time": observed,
                    "matched_model_time": chosen,
                    "hit": 1,
                    "time_offset_h": (chosen - observed).total_seconds() / 3600.0,
                }
            )
        else:
            rows.append(
                {
                    "observed_time": observed,
                    "matched_model_time": pd.NaT,
                    "hit": 0,
                    "time_offset_h": np.nan,
                }
            )
    hits = int(sum(row["hit"] for row in rows))
    counts = {
        "hits": hits,
        "misses": len(observed_times) - hits,
        "false_alarms": len(unmatched_model),
    }
    matched = pd.DataFrame(
        rows,
        columns=["observed_time", "matched_model_time", "hit", "time_offset_h"],
    )
    return counts, matched, unmatched_model


def event_metrics(
    observed: pd.Series,
    modeled: pd.Series,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = pd.DataFrame({"observed": observed, "modeled": modeled}).dropna()
    observed_delta = paired["observed"].diff()
    modeled_delta = paired["modeled"].diff()
    # Never bridge missing timestamps: ramps are defined only for adjacent
    # hourly observations, even if dropna leaves rows on both sides of a gap.
    consecutive_hour = paired.index.to_series().diff().eq(pd.Timedelta(hours=1)).to_numpy()
    observed_delta = observed_delta.where(consecutive_hour)
    modeled_delta = modeled_delta.where(consecutive_hour)
    observed_times = list(paired.index[observed_delta.notna() & observed_delta.abs().ge(threshold)])
    modeled_times = list(paired.index[modeled_delta.notna() & modeled_delta.abs().ge(threshold)])
    observed_meta = pd.DataFrame(
        {
            "event_time": observed_times,
            "ramp_delta": observed_delta.loc[observed_times].to_numpy(float),
        }
    )
    modeled_meta = pd.DataFrame(
        {
            "event_time": modeled_times,
            "ramp_delta": modeled_delta.loc[modeled_times].to_numpy(float),
        }
    )
    observed_meta = observed_meta.reindex(
        columns=["event_time", "ramp_delta"],
    )
    modeled_meta = modeled_meta.reindex(
        columns=["event_time", "ramp_delta"],
    )
    return observed_meta, modeled_meta


def hierarchical_bootstrap_mean(
    frame: pd.DataFrame,
    value_column: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    stations = frame["station_id"].unique()
    station_groups = {
        station: group[value_column].dropna().to_numpy(float)
        for station, group in frame.groupby("station_id")
    }
    estimates = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled_stations = rng.choice(stations, size=len(stations), replace=True)
        station_means = []
        for station in sampled_stations:
            values = station_groups[station]
            if len(values) == 0:
                continue
            sampled_values = rng.choice(values, size=len(values), replace=True)
            station_means.append(float(sampled_values.mean()))
        estimates[iteration] = float(np.mean(station_means))
    return (
        float(frame[value_column].mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def add_direction_metrics(
    row: dict,
    observed_events: pd.DataFrame,
    modeled_events: pd.DataFrame,
    tolerance_h: int,
) -> dict:
    for direction, observed_mask, modeled_mask in (
        ("up", observed_events["ramp_delta"].ge(0), modeled_events["ramp_delta"].ge(0)),
        ("down", observed_events["ramp_delta"].lt(0), modeled_events["ramp_delta"].lt(0)),
    ):
        observed_times = list(observed_events.loc[observed_mask, "event_time"])
        modeled_times = list(modeled_events.loc[modeled_mask, "event_time"])
        counts, _, _ = match_events_detailed(observed_times, modeled_times, tolerance_h)
        row[f"{direction}_obs_events"] = len(observed_times)
        row[f"{direction}_model_events"] = len(modeled_times)
        row[f"{direction}_hits"] = counts["hits"]
        row[f"{direction}_misses"] = counts["misses"]
        row[f"{direction}_false_alarms"] = counts["false_alarms"]
        row[f"{direction}_pod"] = safe_div(
            counts["hits"], counts["hits"] + counts["misses"]
        )
        row[f"{direction}_far"] = safe_div(
            counts["false_alarms"], counts["hits"] + counts["false_alarms"]
        )
        row[f"{direction}_csi"] = safe_div(
            counts["hits"],
            counts["hits"] + counts["misses"] + counts["false_alarms"],
        )
    return row


def load_field(
    spec: dict,
    month: str,
    data_root: Path,
    cache: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_key = (spec["source"], month)
    if cache_key not in cache:
        if spec["source"] == "tianji":
            path = data_root / "processed/tianji_monthly" / month / "DSWRFsfc.nc"
        else:
            folder = ERA5_FOLDERS[month]
            product_folder = {
                "era5_native": "era5_monthly",
                "era5_regridded": "era5_on_tianji_monthly",
            }[spec["source"]]
            path = data_root / "processed" / product_folder / folder / "DSWRFsfc.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as ds:
            variable = "DSWRFsfc" if "DSWRFsfc" in ds.data_vars else next(iter(ds.data_vars))
            field = ds[variable].load()
            times = pd.to_datetime(field["time"].values) + pd.Timedelta(hours=8)
            cache[cache_key] = (
                field.values.astype(float),
                times,
                field["grid_xt"].values,
                field["grid_yt"].values,
            )
    return cache[cache_key]


def build_summary(
    details: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    group_columns = [
        "product",
        "physics_family",
        "source",
        "nominal_resolution_km",
        "support_km",
        "solar_qc_group",
        "threshold_wm2h",
        "tolerance_h",
    ]
    rows = []
    for keys, group in details.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "stations": int(group["station_id"].nunique()),
                "station_months": int(len(group)),
                "paired_hours_mean": float(group["paired_hours"].mean()),
                "obs_events_sum": int(group["obs_events"].sum()),
                "model_events_sum": int(group["model_events"].sum()),
                "hits_sum": int(group["hits"].sum()),
                "misses_sum": int(group["misses"].sum()),
                "false_alarms_sum": int(group["false_alarms"].sum()),
                "csi_mean": float(group["csi"].mean()),
                "pod_mean": float(group["pod"].mean()),
                "far_mean": float(group["far"].mean()),
                "frequency_bias_mean": float(group["frequency_bias"].mean()),
                "up_csi_mean": float(group["up_csi"].mean()),
                "down_csi_mean": float(group["down_csi"].mean()),
                "ghi_corr_mean": float(group["ghi_corr"].mean()),
                "ghi_rmse_mean": float(group["ghi_rmse_wm2"].mean()),
            }
        )
        mean, low, high = hierarchical_bootstrap_mean(
            group,
            "csi",
            n_bootstrap=n_bootstrap,
            seed=seed + len(rows),
        )
        row.update({"csi_ci95_low": low, "csi_ci95_high": high})
        rows.append(row)
    return pd.DataFrame(rows)


def build_resolution_effects(
    details: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for (threshold, tolerance, qc_group), group in details.groupby(
        ["threshold_wm2h", "tolerance_h", "solar_qc_group"]
    ):
        pivot = group.pivot_table(
            index=["station_id", "month"],
            columns="product",
            values="csi",
        )
        if not {"Tianji_2.5km", "Tianji_27.5km"}.issubset(pivot.columns):
            continue
        delta = (pivot["Tianji_27.5km"] - pivot["Tianji_2.5km"]).dropna()
        delta_frame = (
            delta.rename("delta_csi")
            .reset_index()
            .assign(
                station_id=lambda frame: frame["station_id"].astype(str),
            )
        )
        mean, low, high = hierarchical_bootstrap_mean(
            delta_frame,
            "delta_csi",
            n_bootstrap=n_bootstrap,
            seed=seed + len(rows) + 1000,
        )
        rows.append(
            {
                "contrast": "Tianji_27.5km_minus_Tianji_2.5km",
                "threshold_wm2h": threshold,
                "tolerance_h": tolerance,
                "solar_qc_group": qc_group,
                "paired_station_months": len(delta),
                "mean_2.5km_csi": float(pivot["Tianji_2.5km"].mean()),
                "mean_27.5km_csi": float(pivot["Tianji_27.5km"].mean()),
                "mean_delta_csi": mean,
                "ci95_low": low,
                "ci95_high": high,
                "fraction_station_months_with_loss": float((delta < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--months", nargs="+", default=["2025-04", "2025-07", "2025-10"])
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0],
    )
    parser.add_argument("--tolerances", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    stations = pd.read_csv(args.stations, low_memory=False)
    stations["local_time"] = parse_local_datetime(
        stations["datetime_hubei_local_fixed"]
    ).dt.tz_localize(None)
    stations = stations[
        stations["common_period_flag"].eq(1)
        & stations["usable_absolute_event_pairing"].eq(1)
    ].copy()
    if stations.empty:
        raise RuntimeError("No usable station rows after fixing QC/time conventions.")

    metric_rows = []
    observed_event_rows = []
    modeled_event_rows = []
    field_cache: dict = {}

    for product, spec in PRODUCT_SPECS.items():
        for month in args.months:
            try:
                values, model_times, lon_values, lat_values = load_field(
                    spec,
                    month,
                    data_root,
                    field_cache,
                )
            except (FileNotFoundError, KeyError):
                print(f"[SKIP] {product} {month}: field not available")
                continue

            station_month = stations[
                stations["local_time"].dt.strftime("%Y-%m") == month
            ].copy()
            if station_month.empty:
                continue

            for station_id, station_group in station_month.groupby("station_id"):
                station_group = station_group.sort_values("local_time")
                lon = float(station_group["hubei_lon"].iloc[0])
                lat = float(station_group["hubei_lat"].iloc[0])
                site = station_group["hubei_site"].iloc[0]
                qc_group = station_group["solar_qc_group"].iloc[0]

                observed = pd.Series(
                    station_group["ghi_wm2"].to_numpy(float),
                    index=pd.DatetimeIndex(station_group["local_time"]),
                    name="observed",
                )
                if spec["factor"] == 1:
                    field = xr.DataArray(
                        values,
                        dims=("time", "grid_yt", "grid_xt"),
                        coords={
                            "time": model_times,
                            "grid_yt": lat_values,
                            "grid_xt": lon_values,
                        },
                    )
                    modeled_values = sample_footprint(field, lon, lat)
                else:
                    modeled_values = sample_block_mean(
                        values,
                        lon_values,
                        lat_values,
                        lon,
                        lat,
                        int(spec["factor"]),
                    )
                modeled = pd.Series(
                    modeled_values,
                    index=pd.DatetimeIndex(model_times),
                    name="modeled",
                )
                paired = pd.DataFrame(
                    {"observed": observed, "modeled": modeled}
                ).dropna()
                if paired.empty:
                    continue

                error = paired["modeled"] - paired["observed"]
                base = {
                    "product": product,
                    "physics_family": spec["physics_family"],
                    "source": spec["source"],
                    "nominal_resolution_km": spec["nominal_resolution_km"],
                    "support_km": spec["support_km"],
                    "month": month,
                    "station_id": station_id,
                    "hubei_site": site,
                    "solar_qc_group": qc_group,
                    "paired_hours": len(paired),
                    "ghi_mbe_wm2": float(error.mean()),
                    "ghi_mae_wm2": float(error.abs().mean()),
                    "ghi_rmse_wm2": float(np.sqrt((error**2).mean())),
                    "ghi_corr": float(paired["observed"].corr(paired["modeled"])),
                }

                for threshold in args.thresholds:
                    observed_events, modeled_events = event_metrics(
                        paired["observed"],
                        paired["modeled"],
                        threshold,
                    )
                    for tolerance_h in args.tolerances:
                        observed_times = list(observed_events["event_time"])
                        modeled_times = list(modeled_events["event_time"])
                        counts, matched, false_alarms = match_events_detailed(
                            observed_times,
                            modeled_times,
                            tolerance_h,
                        )
                        row = {
                            **base,
                            "threshold_wm2h": threshold,
                            "tolerance_h": tolerance_h,
                            "obs_events": len(observed_times),
                            "model_events": len(modeled_times),
                            **counts,
                            "pod": safe_div(
                                counts["hits"], counts["hits"] + counts["misses"]
                            ),
                            "far": safe_div(
                                counts["false_alarms"],
                                counts["hits"] + counts["false_alarms"],
                            ),
                            "csi": safe_div(
                                counts["hits"],
                                counts["hits"] + counts["misses"] + counts["false_alarms"],
                            ),
                            "frequency_bias": safe_div(
                                len(modeled_times), len(observed_times)
                            ),
                        }
                        row = add_direction_metrics(
                            row,
                            observed_events,
                            modeled_events,
                            tolerance_h,
                        )
                        metric_rows.append(row)

                        if (
                            product
                            in {
                                "Tianji_2.5km",
                                "Tianji_27.5km",
                                "ERA5_native_27.5km",
                                "ERA5_regridded_2.5km",
                            }
                            and threshold in {150.0, 200.0, 250.0}
                            and tolerance_h in {1, 2}
                        ):
                            event_key = pd.concat(
                                [
                                    observed_events.reset_index(drop=True),
                                    matched.reset_index(drop=True),
                                ],
                                axis=1,
                            )
                            for event in event_key.itertuples(index=False):
                                observed_event_rows.append(
                                    {
                                        **base,
                                        "threshold_wm2h": threshold,
                                        "tolerance_h": tolerance_h,
                                        "event_key": (
                                            f"{product}|{station_id}|"
                                            f"{pd.Timestamp(event.event_time).isoformat()}"
                                        ),
                                        "event_time": event.event_time,
                                        "start_hour": pd.Timestamp(event.event_time).hour,
                                        "season": month,
                                        "direction": (
                                            "up" if event.ramp_delta >= 0 else "down"
                                        ),
                                        "ramp_delta": event.ramp_delta,
                                        "hit": int(event.hit),
                                        "time_offset_h": event.time_offset_h,
                                    }
                                )
                            for event_time in false_alarms:
                                modeled_event_rows.append(
                                    {
                                        **base,
                                        "threshold_wm2h": threshold,
                                        "tolerance_h": tolerance_h,
                                        "event_key": (
                                            f"{product}|{station_id}|"
                                            f"{pd.Timestamp(event_time).isoformat()}"
                                        ),
                                        "event_time": event_time,
                                        "start_hour": pd.Timestamp(event_time).hour,
                                        "season": month,
                                        "direction": (
                                            "up"
                                            if modeled_events.loc[
                                                modeled_events["event_time"].eq(event_time),
                                                "ramp_delta",
                                            ].iloc[0]
                                            >= 0
                                            else "down"
                                        ),
                                        "ramp_delta": float(
                                            modeled_events.loc[
                                                modeled_events["event_time"].eq(event_time),
                                                "ramp_delta",
                                            ].iloc[0]
                                        ),
                                        "false_alarm": 1,
                                    }
                                )
            print(f"[OK] {product} {month}", flush=True)

    details = pd.DataFrame(metric_rows)
    details.to_csv(
        output_dir / "canonical_station_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = build_summary(
        details,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    summary.to_csv(
        output_dir / "canonical_main_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    effects = build_resolution_effects(
        details,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    effects.to_csv(
        output_dir / "canonical_resolution_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(observed_event_rows).to_csv(
        output_dir / "event_detection_details.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(modeled_event_rows).to_csv(
        output_dir / "event_false_alarm_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "stations_file": str(Path(args.stations).resolve()),
        "data_root": str(data_root.resolve()),
        "months": args.months,
        "thresholds_wm2h": args.thresholds,
        "tolerances_h": args.tolerances,
        "products": PRODUCT_SPECS,
        "primary_threshold_wm2h": 200.0,
        "primary_tolerance_h": 1,
        "best_case_tolerance_h": 2,
        "station_timestamp_left_as_repaired": True,
        "event_detection_interval": "Only differences between paired rows exactly one hour apart are eligible; missing-hour gaps are not bridged.",
        "bias_correction": "none",
        "spatial_matching": "nearest station cell for all products",
        "bootstrap": "two-stage station-month hierarchical bootstrap",
        "rows": int(len(details)),
    }
    (output_dir / "canonical_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] wrote {output_dir / 'canonical_station_metrics.csv'}")
    print(f"[OK] wrote {output_dir / 'canonical_main_results.csv'}")
    print(f"[OK] wrote {output_dir / 'canonical_resolution_effects.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
