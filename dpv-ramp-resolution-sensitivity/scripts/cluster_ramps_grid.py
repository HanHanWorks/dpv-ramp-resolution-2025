#!/usr/bin/env python3
"""Cluster gridded ramp events with ST-DBSCAN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def add_cluster_area_stats(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clustered = events[events["cluster_id"] >= 0]
    for cluster_id, group in clustered.groupby("cluster_id"):
        lons = group["lon"].to_numpy()
        lats = group["lat"].to_numpy()
        times = pd.to_datetime(group["time"])
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "threshold": float(group["threshold"].iloc[0]),
                "ramp_type_mode": group["ramp_type"].mode().iloc[0],
                "event_count": len(group),
                "time_start": times.min().strftime("%Y-%m-%d %H:%M:%S"),
                "time_end": times.max().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_h": (times.max() - times.min()).total_seconds() / 3600.0 + 1.0,
                "lon_min": float(lons.min()),
                "lon_max": float(lons.max()),
                "lat_min": float(lats.min()),
                "lat_max": float(lats.max()),
                "approx_area_km2": max(1.0, (lons.max() - lons.min()) * 111.0 * np.cos(np.deg2rad(lats.mean()))) * max(1.0, (lats.max() - lats.min()) * 111.0),
                "peak_abs_delta_p": float(group["delta_p"].abs().max()),
                "mean_delta_p": float(group["delta_p"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--events-output", required=True)
    parser.add_argument("--clusters-output", required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--eps-space-km", type=float, default=5.0)
    parser.add_argument("--eps-time-h", type=float, default=1.0)
    parser.add_argument("--min-samples", type=int, default=5)
    args = parser.parse_args()

    events_path = Path(args.events_output)
    clusters_path = Path(args.clusters_output)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    clusters_path.parent.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.input)
    events = events[np.isclose(events["threshold"], args.threshold)].copy()
    if events.empty:
        events["cluster_id"] = pd.Series(dtype=int)
        events.to_csv(events_path, index=False)
        pd.DataFrame().to_csv(clusters_path, index=False)
        print("[WARN] no events to cluster")
        return

    events["cluster_id"] = -1
    next_cluster_id = 0
    for ramp_type, group in events.groupby("ramp_type", sort=True):
        t0 = pd.to_datetime(group["time"]).min()
        time_h = (pd.to_datetime(group["time"]) - t0).dt.total_seconds().to_numpy() / 3600.0
        mean_lat = np.deg2rad(group["lat"].mean())
        x_km = group["lon"].to_numpy() * 111.0 * np.cos(mean_lat)
        y_km = group["lat"].to_numpy() * 111.0
        features = np.column_stack([x_km / args.eps_space_km, y_km / args.eps_space_km, time_h / args.eps_time_h])

        labels = DBSCAN(eps=1.0, min_samples=args.min_samples, n_jobs=-1).fit_predict(features)
        mapped_labels = np.where(labels >= 0, labels + next_cluster_id, -1)
        events.loc[group.index, "cluster_id"] = mapped_labels
        if labels.max(initial=-1) >= 0:
            next_cluster_id += int(labels.max()) + 1

    clusters = add_cluster_area_stats(events)

    events.to_csv(events_path, index=False, encoding="utf-8")
    clusters.to_csv(clusters_path, index=False, encoding="utf-8")
    print(f"[OK] wrote {events_path}")
    print(f"[OK] wrote {clusters_path}")
    print(f"clusters: {(clusters.shape[0])}, noise events: {int((events['cluster_id'] == -1).sum())}, total events: {len(events)}")


if __name__ == "__main__":
    main()
