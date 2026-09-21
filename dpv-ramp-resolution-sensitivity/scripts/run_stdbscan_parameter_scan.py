#!/usr/bin/env python3
"""Scan ST-DBSCAN parameters for robustness reporting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def cluster_events(events: pd.DataFrame, eps_space_km: float, eps_time_h: float, min_samples: int) -> pd.DataFrame:
    if events.empty:
        return events.assign(cluster_id=pd.Series(dtype=int))
    out = events.copy()
    out["cluster_id"] = -1
    next_cluster_id = 0
    for _, group in out.groupby("ramp_type", sort=True):
        t0 = pd.to_datetime(group["time"]).min()
        time_h = (pd.to_datetime(group["time"]) - t0).dt.total_seconds().to_numpy() / 3600.0
        mean_lat = np.deg2rad(group["lat"].mean())
        x_km = group["lon"].to_numpy() * 111.0 * np.cos(mean_lat)
        y_km = group["lat"].to_numpy() * 111.0
        features = np.column_stack([x_km / eps_space_km, y_km / eps_space_km, time_h / eps_time_h])
        labels = DBSCAN(eps=1.0, min_samples=min_samples, n_jobs=-1).fit_predict(features)
        mapped = np.where(labels >= 0, labels + next_cluster_id, -1)
        out.loc[group.index, "cluster_id"] = mapped
        if labels.max(initial=-1) >= 0:
            next_cluster_id += int(labels.max()) + 1
    return out


def cluster_area_stats(clustered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, group in clustered[clustered["cluster_id"] >= 0].groupby("cluster_id"):
        lons = group["lon"].to_numpy()
        lats = group["lat"].to_numpy()
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "event_count": len(group),
                "approx_area_km2": max(1.0, (lons.max() - lons.min()) * 111.0 * np.cos(np.deg2rad(lats.mean())))
                * max(1.0, (lats.max() - lats.min()) * 111.0),
                "peak_abs_delta_p": float(group["delta_p"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/ramp_events/2025-07/ramp_events.csv")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--eps-space-km", nargs="+", type=float, default=[3, 5, 8])
    parser.add_argument("--eps-time-h", nargs="+", type=float, default=[1, 2])
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--output-dir", default="data/analysis/stdbscan_sensitivity")
    parser.add_argument("--report", default="reports/stdbscan_parameter_sensitivity.md")
    args = parser.parse_args()

    events = pd.read_csv(args.input)
    events = events[np.isclose(events["threshold"], args.threshold)].copy()
    rows = []
    for eps_space in args.eps_space_km:
        for eps_time in args.eps_time_h:
            clustered = cluster_events(events, eps_space, eps_time, args.min_samples)
            clusters = cluster_area_stats(clustered)
            noise_events = int((clustered["cluster_id"] < 0).sum())
            rows.append(
                {
                    "eps_space_km": eps_space,
                    "eps_time_h": eps_time,
                    "min_samples": args.min_samples,
                    "event_count": int(len(clustered)),
                    "cluster_count": int(len(clusters)),
                    "noise_event_count": noise_events,
                    "noise_ratio": float(noise_events / max(1, len(clustered))),
                    "median_cluster_area_km2": float(clusters["approx_area_km2"].median()) if not clusters.empty else np.nan,
                    "max_cluster_area_km2": float(clusters["approx_area_km2"].max()) if not clusters.empty else np.nan,
                    "median_cluster_events": float(clusters["event_count"].median()) if not clusters.empty else np.nan,
                    "max_cluster_events": int(clusters["event_count"].max()) if not clusters.empty else 0,
                }
            )
            print(f"[OK] eps_space={eps_space} eps_time={eps_time}")

    summary = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "stdbscan_parameter_scan.csv", index=False, encoding="utf-8")

    baseline = summary[(summary["eps_space_km"].eq(5)) & (summary["eps_time_h"].eq(1))]
    lines = [
        "# ST-DBSCAN parameter sensitivity",
        "",
        f"- Input: `{args.input}`",
        f"- Threshold: {args.threshold} p.u./h",
        f"- min_samples: {args.min_samples}",
        "",
        "## Scan Summary",
        summary.to_markdown(index=False),
        "",
        "## Recommended Setting",
    ]
    if not baseline.empty:
        row = baseline.iloc[0]
        lines.append(
            f"- Use eps_space=5 km and eps_time=1 h as the main setting; it yields {int(row['cluster_count'])} clusters, "
            f"noise ratio {row['noise_ratio']:.3f}, median area {row['median_cluster_area_km2']:.1f} km2."
        )
    else:
        lines.append("- Baseline setting not found in scan.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "- Use this table as robustness evidence for cluster-scale conclusions, not as a physical cloud-object validation.",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out / 'stdbscan_parameter_scan.csv'}")
    print(f"[OK] wrote {args.report}")


if __name__ == "__main__":
    main()
