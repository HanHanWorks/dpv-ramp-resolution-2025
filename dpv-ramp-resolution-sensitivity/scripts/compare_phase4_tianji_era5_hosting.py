#!/usr/bin/env python3
"""Compare Tianji and ERA5-regridded IEEE33 hosting-capacity outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, seed: int, samples: int = 10000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def capped_mask(df: pd.DataFrame) -> pd.Series:
    if "hosting_capacity_label" not in df:
        return pd.Series(False, index=df.index)
    return df["hosting_capacity_label"].astype(str).str.startswith(">=")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tianji", default="data/analysis/ieee33_application_tianji/hosting_capacity_summary.csv")
    parser.add_argument("--era5", default="data/analysis/ieee33_application_era5/hosting_capacity_summary.csv")
    parser.add_argument("--output-dir", default="data/analysis/ieee33_application_delta")
    parser.add_argument("--report", default="reports/ieee33_application_delta.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tianji = pd.read_csv(args.tianji)
    era5 = pd.read_csv(args.era5)
    tianji = tianji.add_prefix("tianji_")
    era5 = era5.add_prefix("era5_")
    merged = tianji.merge(era5, left_on="tianji_scenario_id", right_on="era5_scenario_id", how="inner")
    if merged.empty:
        # ERA5 scenario IDs should normally match Tianji; fall back to rank-by-type
        # pairing if the scenario catalog has been regenerated independently.
        tianji["pair_rank"] = tianji.groupby("tianji_scenario_type").cumcount()
        era5["pair_rank"] = era5.groupby("era5_scenario_type").cumcount()
        merged = tianji.merge(
            era5,
            left_on=["tianji_scenario_type", "pair_rank"],
            right_on=["era5_scenario_type", "pair_rank"],
            how="inner",
        )

    merged["delta_hc_mw"] = merged["tianji_hosting_capacity_mw"] - merged["era5_hosting_capacity_mw"]
    merged["relative_delta_hc_pct"] = merged["delta_hc_mw"] / merged["era5_hosting_capacity_mw"].replace(0, np.nan) * 100.0
    merged["any_capped"] = capped_mask(merged.rename(columns={"tianji_hosting_capacity_label": "hosting_capacity_label"})) | capped_mask(
        merged.rename(columns={"era5_hosting_capacity_label": "hosting_capacity_label"})
    )

    usable = merged[~merged["any_capped"]].copy()
    hc_ci = bootstrap_ci(usable["delta_hc_mw"].to_numpy(dtype=float), args.seed)
    summary = pd.DataFrame(
        [
            {
                "paired_scenarios": int(len(merged)),
                "uncapped_pairs": int(len(usable)),
                "mean_delta_hc_mw": float(usable["delta_hc_mw"].mean()) if not usable.empty else np.nan,
                "delta_hc_ci95_low": hc_ci[0],
                "delta_hc_ci95_high": hc_ci[1],
                "mean_relative_delta_hc_pct": float(usable["relative_delta_hc_pct"].mean()) if not usable.empty else np.nan,
            }
        ]
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "paired_hosting_delta.csv", index=False, encoding="utf-8")
    summary.to_csv(out / "hosting_delta_summary.csv", index=False, encoding="utf-8")

    lines = [
        "# Tianji vs ERA5-regridded IEEE33 hosting-capacity delta",
        "",
        "## Summary",
        summary.to_markdown(index=False),
        "",
        "## Paired Scenario Deltas",
        merged[
            [
                "tianji_scenario_id",
                "tianji_scenario_type",
                "tianji_hosting_capacity_mw",
                "era5_hosting_capacity_mw",
                "delta_hc_mw",
                "relative_delta_hc_pct",
                "tianji_reverse_power_hours",
                "era5_reverse_power_hours",
                "tianji_max_branch_current_pu",
                "era5_max_branch_current_pu",
                "any_capped",
            ]
        ].to_markdown(index=False),
        "",
        "## Claim Boundary",
        "- This is an IEEE33 technical application demonstration, not real-feeder planning.",
        "- If any pair is capped at the scan upper bound, exclude it from strong mean/CI claims.",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out / 'paired_hosting_delta.csv'}")
    print(f"[OK] wrote {out / 'hosting_delta_summary.csv'}")
    print(f"[OK] wrote {args.report}")


if __name__ == "__main__":
    main()
