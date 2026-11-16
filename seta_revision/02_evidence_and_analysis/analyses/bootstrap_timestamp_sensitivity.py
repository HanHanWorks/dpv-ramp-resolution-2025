#!/usr/bin/env python3
"""Paired station-cluster bootstrap for hourly timestamp sensitivity results."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "timestamp_hourly_alignment_sensitivity.csv"
MASK = HERE / "timestamp_hourly_alignment_common_mask.csv"
OUTPUT = HERE / "timestamp_hourly_alignment_station_bootstrap.csv"
N_BOOT = 50000
SEED = 261112
PRODUCTS = [
    "Tianji_2.5km",
    "Tianji_5km",
    "Tianji_10km",
    "Tianji_25km",
    "Tianji_27.5km",
    "ERA5_native_27.5km",
    "ERA5_regridded_2.5km",
]
SHIFTS = [-1, 0, 1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ci(values):
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main():
    frame = pd.read_csv(INPUT)
    mask = pd.read_csv(MASK)
    rows = []
    rng = np.random.default_rng(SEED)

    for qc_group, group in frame.groupby("solar_qc_group"):
        stations = sorted(group["station_id"].unique())
        months = sorted(group["month"].unique())
        expected = pd.MultiIndex.from_product([stations, months], names=["station_id", "month"])
        scores = group.pivot(index=["station_id", "month"], columns=["product", "model_timestamp_shift_h"], values="csi").reindex(expected)
        if scores.isna().any().any():
            raise RuntimeError("Incomplete paired station-month grid for {}".format(qc_group))
        tensor = np.empty((len(stations), len(months), len(PRODUCTS), len(SHIFTS)), dtype=float)
        for pi, product in enumerate(PRODUCTS):
            for si, shift in enumerate(SHIFTS):
                tensor[:, :, pi, si] = scores[(product, shift)].to_numpy().reshape(len(stations), len(months))

        point = tensor.mean(axis=(0, 1))
        boot_abs = np.empty((N_BOOT, len(PRODUCTS), len(SHIFTS)), dtype=float)
        boot_delta = np.empty((N_BOOT, len(PRODUCTS), 2), dtype=float)
        boot_contrast = np.empty((N_BOOT, len(SHIFTS)), dtype=float)
        coarse_idx = PRODUCTS.index("Tianji_27.5km")
        fine_idx = PRODUCTS.index("Tianji_2.5km")

        done = 0
        chunk = 500
        while done < N_BOOT:
            n = min(chunk, N_BOOT - done)
            draw = rng.integers(0, len(stations), size=(n, len(stations)))
            # Resample station IDs as clusters; retain both April and July rows.
            sampled = tensor[draw]  # [draw, sampled station, month, product, shift]
            mean = sampled.mean(axis=(1, 2))
            boot_abs[done : done + n] = mean
            boot_delta[done : done + n, :, 0] = mean[:, :, 0] - mean[:, :, 1]
            boot_delta[done : done + n, :, 1] = mean[:, :, 2] - mean[:, :, 1]
            boot_contrast[done : done + n] = mean[:, coarse_idx, :] - mean[:, fine_idx, :]
            done += n

        for pi, product in enumerate(PRODUCTS):
            for si, shift in enumerate(SHIFTS):
                lo, hi = ci(boot_abs[:, pi, si])
                rows.append(
                    {
                        "qc_group": qc_group,
                        "estimand": "macro_station_month_CSI",
                        "product_or_contrast": product,
                        "shift_h": shift,
                        "comparison_shift_h": np.nan,
                        "point_estimate": float(point[pi, si]),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_station_clusters": len(stations),
                        "station_months_per_cluster": len(months),
                        "bootstrap_replicates": N_BOOT,
                    }
                )
            for di, shift in enumerate([-1, 1]):
                lo, hi = ci(boot_delta[:, pi, di])
                point_delta = float(point[pi, SHIFTS.index(shift)] - point[pi, SHIFTS.index(0)])
                rows.append(
                    {
                        "qc_group": qc_group,
                        "estimand": "paired_shift_minus_zero_CSI_change",
                        "product_or_contrast": product,
                        "shift_h": shift,
                        "comparison_shift_h": 0,
                        "point_estimate": point_delta,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_station_clusters": len(stations),
                        "station_months_per_cluster": len(months),
                        "bootstrap_replicates": N_BOOT,
                    }
                )

        for si, shift in enumerate(SHIFTS):
            lo, hi = ci(boot_contrast[:, si])
            rows.append(
                {
                    "qc_group": qc_group,
                    "estimand": "Tianji_27.5km_minus_2.5km_CSI",
                    "product_or_contrast": "Tianji_27.5km_minus_Tianji_2.5km",
                    "shift_h": shift,
                    "comparison_shift_h": np.nan,
                    "point_estimate": float(point[coarse_idx, si] - point[fine_idx, si]),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_station_clusters": len(stations),
                    "station_months_per_cluster": len(months),
                    "bootstrap_replicates": N_BOOT,
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)
    metadata = {
        "method": "paired station-cluster percentile bootstrap; sample station IDs with replacement and keep both April and July rows for each sampled station",
        "replicates": N_BOOT,
        "seed": SEED,
        "primary_and_secondary_qc_separate": True,
        "input_sha256": {
            INPUT.name: sha256(INPUT),
            MASK.name: sha256(MASK),
        },
        "mask_check": {
            "rows": int(len(mask)),
            "months": sorted(mask["month"].unique().tolist()),
            "station_count": int(mask["station_id"].nunique()),
            "min_valid_consecutive_hour_pairs": int(mask["common_valid_consecutive_hour_pairs"].min()),
            "max_valid_consecutive_hour_pairs": int(mask["common_valid_consecutive_hour_pairs"].max()),
            "note": "event script explicitly disallows pairs crossing month boundaries; common_first_hour/last_hour document within-month valid coverage",
        },
        "caveat": "With 7 primary station clusters, percentile intervals are exploratory and do not address spatial dependence among stations or generalize over seasons.",
        "output": OUTPUT.name,
    }
    OUTPUT.with_name("timestamp_hourly_alignment_station_bootstrap_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(result[result["estimand"].eq("Tianji_27.5km_minus_2.5km_CSI")].to_string(index=False))
    print(result[result["estimand"].eq("paired_shift_minus_zero_CSI_change")].to_string(index=False))


if __name__ == "__main__":
    main()
