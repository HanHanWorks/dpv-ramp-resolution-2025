#!/usr/bin/env python3
"""Compare simplified PV model variants for robustness evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[OK] wrote {path}")


def solar_proxy(times: xr.DataArray, lats: xr.DataArray, lons: xr.DataArray) -> xr.DataArray:
    ts = pd.to_datetime(times.values)
    hour = xr.DataArray(ts.hour + ts.minute / 60.0, dims="time", coords={"time": times})
    doy = xr.DataArray(ts.dayofyear, dims="time", coords={"time": times})
    decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    lat_rad = np.deg2rad(lats)
    hour_angle = np.deg2rad(15.0 * (hour - 12.0))
    proxy = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    proxy = proxy.transpose("time", "grid_yt")
    proxy = proxy.expand_dims(grid_xt=lons).transpose("time", "grid_yt", "grid_xt")
    return proxy.clip(min=0, max=1).astype("float32")


def build_models(weather_root: Path, m0_path: Path, output_root: Path) -> dict[str, xr.DataArray]:
    with (
        xr.open_dataset(m0_path) as m0_ds,
        xr.open_dataset(weather_root / "DSWRFsfc.nc") as ghi_ds,
        xr.open_dataset(weather_root / "t2mz.nc") as t_ds,
        xr.open_dataset(weather_root / "UGRD10m.nc") as u_ds,
        xr.open_dataset(weather_root / "VGRD10m.nc") as v_ds,
    ):
        m0 = m0_ds["pv_pu"].load()
        ghi = ghi_ds["DSWRFsfc"].clip(min=0).load()
        tamb = (t_ds["t2mz"] - 273.15).load()
        ws10 = np.sqrt(u_ds["UGRD10m"] ** 2 + v_ds["VGRD10m"] ** 2).load()

    proxy = solar_proxy(ghi["time"], ghi["grid_yt"], ghi["grid_xt"])
    clear_norm = proxy / proxy.max(dim="time").clip(min=0.05)
    m1 = (m0 * (0.82 + 0.18 * clear_norm)).clip(min=0, max=1.2).rename("pv_pu")

    tcell_m2 = tamb + ghi / (30.0 + 8.0 * ws10.clip(min=0))
    m2 = (ghi / 1000.0) * (1 - 0.0042 * (tcell_m2 - 25.0)) * 0.965
    m2 = m2.clip(min=0, max=1.2).rename("pv_pu")

    output_root.mkdir(parents=True, exist_ok=True)
    models = {"M0_current": m0, "M1_solar_proxy": m1.astype("float32"), "M2_temp_wind_sensitive": m2.astype("float32")}
    for name, da in models.items():
        path = output_root / f"{name}.nc"
        xr.Dataset({"pv_pu": da}).to_netcdf(path)
        print(f"[OK] wrote {path}")
    return models


def ramp_summary(models: dict[str, xr.DataArray], terrain: pd.DataFrame, threshold: float) -> pd.DataFrame:
    terrain_idx = terrain.set_index(["grid_y", "grid_x"])["terrain_class"]
    rows = []
    for name, pv in models.items():
        delta = pv.diff("time")
        p_now = pv.isel(time=slice(1, None))
        valid = p_now >= 0.1
        event = (np.abs(delta) >= threshold) & valid
        total = int(event.sum())
        up = int(((delta >= threshold) & valid).sum())
        down = int(((delta <= -threshold) & valid).sum())
        rows.append(
            {
                "model": name,
                "scope": "domain",
                "terrain_class": "all",
                "threshold": threshold,
                "event_count": total,
                "up_events": up,
                "down_events": down,
                "mean_pv": float(pv.mean()),
                "max_pv": float(pv.max()),
                "mean_abs_delta_p": float(np.abs(delta).mean()),
            }
        )
        event_df = event.stack(cell=("grid_yt", "grid_xt")).sum("time").to_pandas().reset_index(name="event_count")
        event_df["grid_y"] = event_df["grid_yt"].map({v: i for i, v in enumerate(pv.grid_yt.values)})
        event_df["grid_x"] = event_df["grid_xt"].map({v: i for i, v in enumerate(pv.grid_xt.values)})
        event_df["terrain_class"] = [terrain_idx.loc[(int(r.grid_y), int(r.grid_x))] for r in event_df.itertuples()]
        for terrain_class, group in event_df.groupby("terrain_class"):
            rows.append(
                {
                    "model": name,
                    "scope": "terrain",
                    "terrain_class": terrain_class,
                    "threshold": threshold,
                    "event_count": int(group["event_count"].sum()),
                    "up_events": np.nan,
                    "down_events": np.nan,
                    "mean_pv": np.nan,
                    "max_pv": np.nan,
                    "mean_abs_delta_p": np.nan,
                    "events_per_grid_cell": float(group["event_count"].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_results(models: dict[str, xr.DataArray], summary: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(9, 4.5))
    for name, pv in models.items():
        pv.mean(dim=("grid_yt", "grid_xt")).isel(time=slice(0, 72)).plot(label=name)
    plt.legend()
    plt.ylabel("PV p.u.")
    plt.title("PV model domain-mean profiles, first 72 hours")
    savefig(fig_dir / "24_pv_model_daily_profiles.png")

    plt.figure(figsize=(8, 4.5))
    for name, pv in models.items():
        delta = pv.diff("time").values.ravel()
        plt.hist(delta[np.isfinite(delta)], bins=120, alpha=0.4, label=name)
    plt.legend()
    plt.xlabel("Hourly delta PV p.u.")
    plt.ylabel("Grid-hour count")
    plt.title("PV model ramp magnitude distribution")
    savefig(fig_dir / "25_pv_model_ramp_distribution.png")

    domain = summary[summary["scope"].eq("domain")]
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(domain))
    plt.bar(x - 0.18, domain["up_events"], width=0.36, label="up")
    plt.bar(x + 0.18, domain["down_events"], width=0.36, label="down")
    plt.xticks(x, domain["model"], rotation=15, ha="right")
    plt.ylabel("Events")
    plt.title("Ramp event counts by PV model")
    plt.legend()
    savefig(fig_dir / "26_pv_model_ramp_counts.png")

    terrain = summary[summary["scope"].eq("terrain")]
    pivot = terrain.pivot(index="terrain_class", columns="model", values="events_per_grid_cell")
    pivot.plot(kind="bar", figsize=(9, 4.8))
    plt.ylabel("Events per grid cell")
    plt.title("Terrain-conditioned ramp frequency by PV model")
    savefig(fig_dir / "27_pv_model_terrain_contrast.png")


def write_report(output: Path, summary: pd.DataFrame, fig_dir: Path) -> None:
    lines = [
        "# PV model comparison report",
        "",
        "## Model definitions",
        "- M0_current: existing simplified GHI-temperature-wind PV model.",
        "- M1_solar_proxy: M0 with latitude/time solar-height proxy modulation.",
        "- M2_temp_wind_sensitive: modified temperature/wind cooling and temperature coefficient.",
        "",
        "## Ramp Summary",
        summary.to_markdown(index=False),
        "",
        "## Limitations",
        "- No measured PV calibration is included.",
        "- These variants are robustness checks, not validated plant models.",
        "",
        "## Figures",
    ]
    lines.extend(f"- {p}" for p in sorted(fig_dir.glob("*.png")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-root", default="data/processed/tianji_monthly/2025-07")
    parser.add_argument("--m0", default="data/processed/pv_grid/2025-07/pv_pu.nc")
    parser.add_argument("--terrain", default="data/processed/terrain/terrain_classes.csv")
    parser.add_argument("--model-output-root", default="data/processed/pv_grid_model_compare/2025-07")
    parser.add_argument("--analysis-output", default="data/analysis/pv_model_compare/pv_model_ramp_summary.csv")
    parser.add_argument("--fig-dir", default="figures/pv_model_compare")
    parser.add_argument("--report", default="reports/pv_model_comparison_report.md")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    models = build_models(Path(args.weather_root), Path(args.m0), Path(args.model_output_root))
    terrain = pd.read_csv(args.terrain)
    summary = ramp_summary(models, terrain, args.threshold)
    output = Path(args.analysis_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False, encoding="utf-8")
    print(f"[OK] wrote {output}")
    fig_dir = Path(args.fig_dir)
    plot_results(models, summary, fig_dir)
    write_report(Path(args.report), summary, fig_dir)


if __name__ == "__main__":
    main()
