#!/usr/bin/env python3
"""Summarize the all-seven-product common-hour timestamp rerun.

Input is the detailed output of rerun_fair_mask_benchmark.py.
Bootstrap units are the seven primary-QC station clusters; each sampled
station retains its April and July observations. Pairs always share the same
station-hour mask.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "timestamp_redetection_sensitivity.csv"
DETAIL_OUT = HERE / "fair_mask_common_month_product_summary.csv"
PAIRED_OUT = HERE / "fair_mask_common_month_paired_contrasts.csv"
META_OUT = HERE / "fair_mask_common_month_metadata.json"
PRODUCTS = [
    "Tianji_2.5km", "Tianji_5km", "Tianji_10km", "Tianji_25km",
    "Tianji_27.5km", "ERA5_native_27.5km", "ERA5_regridded_2.5km",
]
N_BOOT = 50_000
SEED = 261113


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ci_station_cluster(values: pd.Series, sampled_indices: np.ndarray) -> tuple[float, float]:
    station_values = values.groupby(level="station_id", sort=True).mean().to_numpy(float)
    draws = station_values[sampled_indices].mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def main() -> None:
    detail = pd.read_csv(SOURCE)
    detail = detail[
        detail["mask_scope"].eq("all_products_common_hours")
        & detail["solar_qc_group"].eq("primary_solar_qc")
        & detail["station_time_shift_h"].eq(0)
        & detail["month"].isin(["2025-04", "2025-07"])
    ].copy()
    expected = {(product, tol) for product in PRODUCTS for tol in [1, 2]}
    observed = set(zip(detail["product"], detail["tolerance_h"].astype(int)))
    if observed != expected:
        raise RuntimeError(f"Unexpected product/tolerance coverage: {observed ^ expected}")
    coverage = detail.groupby(["product", "tolerance_h"]).size()
    if not coverage.eq(14).all():
        raise RuntimeError(f"Expected 14 station-months per product/tolerance: {coverage}")
    mask_hours = detail.groupby(["station_id", "month"])["paired_hours"].nunique()
    if not mask_hours.eq(1).all():
        raise RuntimeError("All products do not have identical paired-hour counts per station-month")
    all_products = detail.groupby(["station_id", "month"])["product"].nunique()
    if not all_products.eq(len(PRODUCTS)).all():
        raise RuntimeError("A station-month is missing one or more benchmark products")

    rng = np.random.default_rng(SEED)
    station_ids = np.array(sorted(detail.station_id.unique()))
    station_draws = rng.integers(0, len(station_ids), size=(N_BOOT, len(station_ids)))
    summary_rows = []
    contrast_rows = []
    for tol in [1, 2]:
        part = detail[detail.tolerance_h.eq(tol)].copy()
        part["far"] = part.false_alarms / (part.hits + part.false_alarms)
        part["frequency_bias"] = part.model_events / part.obs_events
        for product in PRODUCTS:
            block = part[part["product"].eq(product)].set_index(["station_id", "month"]).sort_index()
            values = {
                "csi": block.csi,
                "pod": block.pod,
                "far": block.far,
                "frequency_bias": block.frequency_bias,
            }
            row = {
                "mask_scope": "all_products_common_hours",
                "month_coverage": "2025-04,2025-07",
                "product": product,
                "tolerance_h": tol,
                "stations": block.index.get_level_values("station_id").nunique(),
                "station_months": len(block),
                "paired_hours_sum": int(block.paired_hours.sum()),
                "paired_hours_per_station_month_min": int(block.paired_hours.min()),
                "paired_hours_per_station_month_max": int(block.paired_hours.max()),
                "observed_events_sum": int(block.obs_events.sum()),
                "model_events_sum": int(block.model_events.sum()),
                "hits_sum": int(block.hits.sum()),
                "misses_sum": int(block.misses.sum()),
                "false_alarms_sum": int(block.false_alarms.sum()),
            }
            for metric, series in values.items():
                lo, hi = ci_station_cluster(series, station_draws)
                row[f"macro_{metric}"] = float(series.mean())
                row[f"{metric}_station_cluster_95_low"] = lo
                row[f"{metric}_station_cluster_95_high"] = hi
            summary_rows.append(row)

        wide = part.pivot(index=["station_id", "month"], columns="product", values="csi").sort_index()
        for product in PRODUCTS:
            if product == "Tianji_2.5km":
                continue
            paired_delta = (wide[product] - wide["Tianji_2.5km"]).groupby(level="station_id").mean()
            station_delta = paired_delta.to_numpy(float)
            draws = station_delta[station_draws].mean(axis=1)
            lo, hi = np.quantile(draws, [0.025, 0.975])
            contrast_rows.append({
                "mask_scope": "all_products_common_hours",
                "month_coverage": "2025-04,2025-07",
                "threshold_wm2h": 200,
                "tolerance_h": tol,
                "contrast": f"{product} minus Tianji_2.5km",
                "mean_paired_station_month_delta_csi": float(paired_delta.mean()),
                "station_cluster_95_low": float(lo),
                "station_cluster_95_high": float(hi),
                "stations": len(station_delta),
                "station_months": 14,
                "bootstrap_replicates": N_BOOT,
                "bootstrap_seed": SEED,
                "bootstrap_unit": "station_id; paired April and July rows retained together",
            })

    pd.DataFrame(summary_rows).to_csv(DETAIL_OUT, index=False, float_format="%.8g")
    pd.DataFrame(contrast_rows).to_csv(PAIRED_OUT, index=False, float_format="%.8g")
    meta = {
        "analysis_type": "common-April/July station-hour-mask benchmark across all seven products",
        "source_detail_csv": str(SOURCE.relative_to(HERE.parents[2])),
        "source_detail_sha256": sha256(SOURCE),
        "source_reanalysis_script": "rerun_fair_mask_benchmark.py",
        "station_source": "paper_writing_package_seta_major_revision/01_data/station_reference/integrated_15site_validation_hourly_repaired.csv",
        "products": PRODUCTS,
        "threshold_wm2h": 200,
        "station_time_shift_h": 0,
        "station_qc_scope": "seven primary-QC stations",
        "station_months": 14,
        "bootstrap": {
            "type": "paired station-cluster percentile bootstrap",
            "unit": "station_id; resample seven stations with replacement and retain both April and July rows",
            "replicates": N_BOOT,
            "seed": SEED,
            "limits": "Seven clusters; exploratory intervals do not resolve spatial dependence or seasonal generalizability.",
        },
        "hour_mask": "For each station-month and timestamp shift, intersect station observations with valid values from all seven products; events are redetected after masking, and only differences between timestamps exactly one hour apart are eligible.",
        "outputs": [DETAIL_OUT.name, PAIRED_OUT.name],
    }
    META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).query("tolerance_h == 1")[[
        "product", "macro_csi", "csi_station_cluster_95_low", "csi_station_cluster_95_high",
        "paired_hours_sum", "observed_events_sum", "model_events_sum", "hits_sum", "misses_sum", "false_alarms_sum"
    ]].to_string(index=False))
    print(pd.DataFrame(contrast_rows).query("tolerance_h == 1").to_string(index=False))


if __name__ == "__main__":
    main()
