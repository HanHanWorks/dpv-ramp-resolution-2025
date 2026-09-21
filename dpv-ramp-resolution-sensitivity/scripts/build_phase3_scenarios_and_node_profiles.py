#!/usr/bin/env python3
"""Build representative scenarios and IEEE-33 node PV profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


TERRAIN_WEIGHTS = {
    "high_relief_high_elevation": 1.30,
    "low_relief_high_elevation": 1.15,
    "high_relief_mid_elevation": 1.05,
    "low_relief_low_elevation": 0.90,
}


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[OK] wrote {path}")


def representative_scenarios(clusters: pd.DataFrame, cluster_events: pd.DataFrame, terrain_events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    used: set[int] = set()

    def add_from_cluster(row: pd.Series, scenario_type: str) -> None:
        cid = int(row["cluster_id"])
        if cid in used:
            return
        used.add(cid)
        ge = cluster_events[cluster_events["cluster_id"].eq(cid)]
        terrain_mix = ge.merge(
            terrain_events[["grid_x", "grid_y", "terrain_class"]].drop_duplicates(),
            on=["grid_x", "grid_y"],
            how="left",
        )
        dominant_terrain = terrain_mix["terrain_class"].mode().iloc[0] if not terrain_mix.empty else ""
        rows.append(
            {
                "scenario_id": f"S{len(rows) + 1:02d}",
                "scenario_type": scenario_type,
                "cluster_id": cid,
                "ramp_type": row["ramp_type_mode"],
                "time_start": row["time_start"],
                "time_end": row["time_end"],
                "window_start": (pd.Timestamp(row["time_start"]) - pd.Timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "window_end": (pd.Timestamp(row["time_end"]) + pd.Timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "event_count": int(row["event_count"]),
                "approx_area_km2": float(row["approx_area_km2"]),
                "peak_abs_delta_p": float(row["peak_abs_delta_p"]),
                "mean_delta_p": float(row["mean_delta_p"]),
                "dominant_terrain_class": dominant_terrain,
                "lon_min": float(row["lon_min"]),
                "lon_max": float(row["lon_max"]),
                "lat_min": float(row["lat_min"]),
                "lat_max": float(row["lat_max"]),
            }
        )

    for _, row in clusters[clusters["ramp_type_mode"].eq("down")].sort_values("event_count", ascending=False).head(5).iterrows():
        add_from_cluster(row, "top_ramp_down")
    for _, row in clusters[clusters["ramp_type_mode"].eq("up")].sort_values("event_count", ascending=False).head(3).iterrows():
        add_from_cluster(row, "top_ramp_up")

    enriched = []
    for _, row in clusters.iterrows():
        ge = cluster_events[cluster_events["cluster_id"].eq(row["cluster_id"])]
        if ge.empty:
            continue
        mix = ge.merge(
            terrain_events[["grid_x", "grid_y", "terrain_class"]].drop_duplicates(),
            on=["grid_x", "grid_y"],
            how="left",
        )
        high_share = mix["terrain_class"].eq("high_relief_high_elevation").mean()
        low_share = mix["terrain_class"].eq("low_relief_low_elevation").mean()
        enriched.append({"cluster_id": int(row["cluster_id"]), "high_share": high_share, "low_share": low_share})
    enriched_df = clusters.merge(pd.DataFrame(enriched), on="cluster_id", how="left").fillna({"high_share": 0, "low_share": 0})

    for _, row in enriched_df.sort_values(["high_share", "event_count"], ascending=False).iterrows():
        if len([r for r in rows if r["scenario_type"] == "high_relief_high_elevation_risk"]) >= 2:
            break
        if row["high_share"] > 0:
            add_from_cluster(row, "high_relief_high_elevation_risk")

    for _, row in enriched_df.sort_values(["low_share", "event_count"], ascending=False).iterrows():
        if len([r for r in rows if r["scenario_type"] == "low_relief_low_elevation_control"]) >= 2:
            break
        if row["low_share"] > 0:
            add_from_cluster(row, "low_relief_low_elevation_control")

    return pd.DataFrame(rows)


def build_node_map(terrain: pd.DataFrame) -> pd.DataFrame:
    lon_min, lon_max = terrain["lon"].min(), terrain["lon"].max()
    lat_min, lat_max = terrain["lat"].min(), terrain["lat"].max()
    lon_bins = np.linspace(lon_min - 1e-9, lon_max + 1e-9, 12)
    lat_bins = np.linspace(lat_min - 1e-9, lat_max + 1e-9, 4)
    out = terrain.copy()
    col = np.clip(np.digitize(out["lon"], lon_bins) - 1, 0, 10)
    row = np.clip(np.digitize(out["lat"], lat_bins) - 1, 0, 2)
    out["ieee33_node"] = (row * 11 + col + 1).astype(int)
    out["terrain_weight"] = out["terrain_class"].map(TERRAIN_WEIGHTS).fillna(1.0)
    return out[
        [
            "grid_x",
            "grid_y",
            "lon",
            "lat",
            "terrain_class",
            "elevation_m",
            "slope_deg",
            "tri_m",
            "relief_5x5_m",
            "ieee33_node",
            "terrain_weight",
        ]
    ]


def node_profiles(pv_path: str, node_map: pd.DataFrame) -> pd.DataFrame:
    weighted = node_map[["grid_x", "grid_y", "ieee33_node", "terrain_weight"]].copy()
    weighted["capacity_weight"] = weighted["terrain_weight"] / weighted["terrain_weight"].sum()
    node_capacity_share = weighted.groupby("ieee33_node")["capacity_weight"].sum()

    rows = []
    with xr.open_dataset(pv_path) as ds:
        pv = ds["pv_pu"].load()
        for node, group in weighted.groupby("ieee33_node"):
            node_da = pv.isel(grid_xt=xr.DataArray(group["grid_x"].values, dims="cell"), grid_yt=xr.DataArray(group["grid_y"].values, dims="cell"))
            weights = xr.DataArray(group["terrain_weight"].values, dims="cell")
            profile = node_da.weighted(weights).mean(dim="cell").to_series()
            for time, pv_pu in profile.items():
                rows.append(
                    {
                        "time": pd.Timestamp(time).strftime("%Y-%m-%d %H:%M:%S"),
                        "ieee33_node": int(node),
                        "pv_pu": float(pv_pu),
                        "capacity_share": float(node_capacity_share.loc[node]),
                    }
                )
    return pd.DataFrame(rows)


def plot_scenarios(scenarios: pd.DataFrame, node_map: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(node_map["lon"], node_map["lat"], c=node_map["ieee33_node"], s=5, cmap="tab20", alpha=0.55)
    plt.colorbar(sc, label="IEEE-33 node service area")
    for _, row in scenarios.iterrows():
        plt.gca().add_patch(
            plt.Rectangle(
                (row["lon_min"], row["lat_min"]),
                row["lon_max"] - row["lon_min"],
                row["lat_max"] - row["lat_min"],
                fill=False,
                linewidth=1.2,
                edgecolor="black",
            )
        )
        plt.text(row["lon_min"], row["lat_max"], row["scenario_id"], fontsize=8)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Scenario catalog and IEEE-33 spatial node mapping")
    savefig(fig_dir / "16_scenario_catalog_map.png")


def plot_node_profiles(scenarios: pd.DataFrame, profiles: pd.DataFrame, fig_dir: Path) -> None:
    sample = scenarios.head(4)
    fig, axes = plt.subplots(len(sample), 1, figsize=(10, 2.5 * len(sample)), sharex=False)
    if len(sample) == 1:
        axes = [axes]
    for ax, (_, scenario) in zip(axes, sample.iterrows()):
        window = profiles[
            (pd.to_datetime(profiles["time"]) >= pd.Timestamp(scenario["window_start"]))
            & (pd.to_datetime(profiles["time"]) <= pd.Timestamp(scenario["window_end"]))
        ]
        top_nodes = window.groupby("ieee33_node")["pv_pu"].max().sort_values(ascending=False).head(5).index
        for node in top_nodes:
            group = window[window["ieee33_node"].eq(node)]
            ax.plot(pd.to_datetime(group["time"]), group["pv_pu"], marker="o", label=f"N{node}")
        ax.set_title(f"{scenario['scenario_id']} {scenario['scenario_type']} cluster={scenario['cluster_id']}")
        ax.set_ylabel("PV p.u.")
        ax.legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("Time")
    savefig(fig_dir / "17_node_pv_profiles_typical_scenarios.png")


def write_report(output: Path, scenarios: pd.DataFrame, node_map: pd.DataFrame, profiles: pd.DataFrame, fig_dir: Path) -> None:
    lines = [
        "# Phase 3 scenario catalog and node PV aggregation report",
        "",
        f"- Scenarios: {len(scenarios)}",
        f"- Grid cells mapped to IEEE-33 nodes: {len(node_map)}",
        f"- IEEE-33 nodes represented: {node_map['ieee33_node'].nunique()}",
        f"- Node PV profile rows: {len(profiles)}",
        "",
        "## Scenario Catalog",
        scenarios.to_markdown(index=False),
        "",
        "## Node Capacity Shares",
        node_map.groupby("ieee33_node")["terrain_weight"].sum().reset_index(name="terrain_weight_sum").to_markdown(index=False),
        "",
        "## Figures",
    ]
    lines.extend(f"- {p}" for p in sorted(fig_dir.glob("*.png")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pv", default="data/processed/pv_grid/2025-07/pv_pu.nc")
    parser.add_argument("--clusters", default="data/processed/ramp_clusters/2025-07/ramp_clusters.csv")
    parser.add_argument("--cluster-events", default="data/processed/ramp_clusters/2025-07/ramp_events_clustered.csv")
    parser.add_argument("--terrain-classes", default="data/processed/terrain/terrain_classes.csv")
    parser.add_argument("--output-dir", default="data/analysis/phase3")
    parser.add_argument("--fig-dir", default="figures/phase3")
    parser.add_argument("--report", default="reports/phase3_scenarios_node_profiles.md")
    args = parser.parse_args()

    clusters = pd.read_csv(args.clusters)
    cluster_events = pd.read_csv(args.cluster_events, parse_dates=["time"])
    terrain = pd.read_csv(args.terrain_classes)
    terrain_events = cluster_events.merge(terrain[["grid_x", "grid_y", "terrain_class"]], on=["grid_x", "grid_y"], how="left")

    scenarios = representative_scenarios(clusters, cluster_events, terrain_events)
    node_map = build_node_map(terrain)
    profiles = node_profiles(args.pv, node_map)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(output_dir / "scenario_catalog.csv", index=False, encoding="utf-8")
    node_map.to_csv(output_dir / "grid_to_ieee33_node_map.csv", index=False, encoding="utf-8")
    profiles.to_csv(output_dir / "node_pv_profiles.csv", index=False, encoding="utf-8")
    print(f"[OK] wrote {output_dir / 'scenario_catalog.csv'}")
    print(f"[OK] wrote {output_dir / 'grid_to_ieee33_node_map.csv'}")
    print(f"[OK] wrote {output_dir / 'node_pv_profiles.csv'}")

    fig_dir = Path(args.fig_dir)
    plot_scenarios(scenarios, node_map, fig_dir)
    plot_node_profiles(scenarios, profiles, fig_dir)
    write_report(Path(args.report), scenarios, node_map, profiles, fig_dir)


if __name__ == "__main__":
    main()
