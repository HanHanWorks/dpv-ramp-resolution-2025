#!/usr/bin/env python3
"""Run an IEEE-33 hosting-capacity application demonstration.

This public-release script is intentionally limited to technical screening
metrics. It does not implement economic valuation, investment analysis, or
storage sizing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_MVA = 10.0
BASE_KV = 12.66
Z_BASE = BASE_KV**2 / BASE_MVA
V_LIMIT_LOW = 0.95
V_LIMIT_HIGH = 1.05


BRANCHES = [
    (1, 2, 0.0922, 0.0470), (2, 3, 0.4930, 0.2511), (3, 4, 0.3660, 0.1864), (4, 5, 0.3811, 0.1941),
    (5, 6, 0.8190, 0.7070), (6, 7, 0.1872, 0.6188), (7, 8, 1.7114, 1.2351), (8, 9, 1.0300, 0.7400),
    (9, 10, 1.0440, 0.7400), (10, 11, 0.1966, 0.0650), (11, 12, 0.3744, 0.1238), (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129), (14, 15, 0.5910, 0.5260), (15, 16, 0.7463, 0.5450), (16, 17, 1.2890, 1.7210),
    (17, 18, 0.7320, 0.5740), (2, 19, 0.1640, 0.1565), (19, 20, 1.5042, 1.3554), (20, 21, 0.4095, 0.4784),
    (21, 22, 0.7089, 0.9373), (3, 23, 0.4512, 0.3083), (23, 24, 0.8980, 0.7091), (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034), (26, 27, 0.2842, 0.1447), (27, 28, 1.0590, 0.9337), (28, 29, 0.8042, 0.7006),
    (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.9630), (31, 32, 0.3105, 0.3619), (32, 33, 0.3410, 0.5302),
]


LOADS_KW_KVAR = {
    2: (100, 60), 3: (90, 40), 4: (120, 80), 5: (60, 30), 6: (60, 20), 7: (200, 100), 8: (200, 100),
    9: (60, 20), 10: (60, 20), 11: (45, 30), 12: (60, 35), 13: (60, 35), 14: (120, 80), 15: (60, 10),
    16: (60, 20), 17: (60, 20), 18: (90, 40), 19: (90, 40), 20: (90, 40), 21: (90, 40), 22: (90, 40),
    23: (90, 50), 24: (420, 200), 25: (420, 200), 26: (60, 25), 27: (60, 25), 28: (60, 20), 29: (120, 70),
    30: (200, 600), 31: (150, 70), 32: (210, 100), 33: (60, 40),
}


def children_map() -> dict[int, list[tuple[int, complex]]]:
    children: dict[int, list[tuple[int, complex]]] = {i: [] for i in range(1, 34)}
    for parent, child, r, x in BRANCHES:
        children[parent].append((child, complex(r / Z_BASE, x / Z_BASE)))
    return children


def subtree_order() -> list[int]:
    return [child for _, child, _, _ in BRANCHES][::-1]


def load_arrays() -> tuple[np.ndarray, np.ndarray]:
    p = np.zeros(34)
    q = np.zeros(34)
    for node, (kw, kvar) in LOADS_KW_KVAR.items():
        p[node] = kw / 1000.0 / BASE_MVA
        q[node] = kvar / 1000.0 / BASE_MVA
    return p, q


def run_power_flow(pv_mw_by_node: dict[int, float], slack_voltage_pu: float = 1.05, max_iter: int = 100, tol: float = 1e-7) -> dict[str, object]:
    children = children_map()
    p_load, q_load = load_arrays()
    p_net = p_load.copy()
    for node, mw in pv_mw_by_node.items():
        p_net[node] -= mw / BASE_MVA
    s_net = p_net + 1j * q_load
    voltages = np.ones(34, dtype=complex) * slack_voltage_pu
    branch_current: dict[tuple[int, int], complex] = {}
    converged = False
    for _ in range(max_iter):
        old = voltages.copy()
        inj_current = np.conj(s_net / np.where(np.abs(voltages) > 1e-6, voltages, 1.0))
        downstream = inj_current.copy()
        for child in subtree_order():
            parent = next(p for p, c, _, _ in BRANCHES if c == child)
            downstream[parent] += downstream[child]
        voltages[1] = slack_voltage_pu + 0j
        stack = [1]
        while stack:
            parent = stack.pop()
            for child, z in children[parent]:
                branch_current[(parent, child)] = downstream[child]
                voltages[child] = voltages[parent] - z * downstream[child]
                stack.append(child)
        if np.max(np.abs(voltages - old)) < tol:
            converged = True
            break
    vmag = np.abs(voltages[1:])
    currents = np.array([abs(branch_current[(p, c)]) for p, c, _, _ in BRANCHES])
    root_p_mw = float((voltages[1] * np.conj(sum(branch_current[(1, c)] for c, _ in children[1]))).real * BASE_MVA)
    return {
        "converged": converged,
        "voltages": vmag,
        "branch_currents": currents,
        "v_min": float(vmag.min()),
        "v_max": float(vmag.max()),
        "max_branch_current_pu": float(currents.max()),
        "reverse_power_mw": max(0.0, -root_p_mw),
        "root_p_mw": root_p_mw,
    }


def scenario_window_profiles(profiles: pd.DataFrame, scenario: pd.Series) -> pd.DataFrame:
    times = pd.to_datetime(profiles["time"])
    return profiles[(times >= pd.Timestamp(scenario["window_start"])) & (times <= pd.Timestamp(scenario["window_end"]))].copy()


def evaluate_scenario(
    scenario: pd.Series,
    profiles: pd.DataFrame,
    capacities_mw: np.ndarray,
    slack_voltage_pu: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    window = scenario_window_profiles(profiles, scenario)
    rows = []
    summary_at_best: dict[str, object] | None = None
    feasible_caps = []
    for cap in capacities_mw:
        violation_hours = 0
        max_load = 0.0
        reverse_hours = 0
        reverse_energy = 0.0
        pv_energy = 0.0
        vmins = []
        vmaxs = []
        for time, group in window.groupby("time"):
            pv_by_node = {int(r.ieee33_node): float(cap * r.capacity_share * r.pv_pu) for r in group.itertuples()}
            total_pv = sum(pv_by_node.values())
            pv_energy += total_pv
            pf = run_power_flow(pv_by_node, slack_voltage_pu=slack_voltage_pu)
            vmins.append(pf["v_min"])
            vmaxs.append(pf["v_max"])
            max_load = max(max_load, pf["max_branch_current_pu"])
            reverse_energy += pf["reverse_power_mw"]
            if pf["reverse_power_mw"] > 1e-6:
                reverse_hours += 1
            if pf["v_min"] < V_LIMIT_LOW or pf["v_max"] > V_LIMIT_HIGH:
                violation_hours += 1
        feasible = violation_hours == 0
        if feasible:
            feasible_caps.append(cap)
        row = {
            "scenario_id": scenario["scenario_id"],
            "scenario_type": scenario["scenario_type"],
            "cluster_id": int(scenario["cluster_id"]),
            "dpv_capacity_mw": float(cap),
            "hours": int(window["time"].nunique()),
            "v_min": min(vmins) if vmins else np.nan,
            "v_max": max(vmaxs) if vmaxs else np.nan,
            "voltage_violation_hours": int(violation_hours),
            "reverse_power_hours": int(reverse_hours),
            "reverse_energy_mwh": float(reverse_energy),
            "max_branch_current_pu": float(max_load),
            "pv_energy_mwh": float(pv_energy),
            "feasible": feasible,
        }
        rows.append(row)
    best = max(feasible_caps) if feasible_caps else 0.0
    best_row = max([r for r in rows if r["dpv_capacity_mw"] == best], key=lambda r: r["dpv_capacity_mw"])
    summary_at_best = best_row.copy()
    summary_at_best["hosting_capacity_mw"] = best
    summary_at_best["hosting_capacity_label"] = f">={capacities_mw.max():.2f}" if best >= capacities_mw.max() else f"{best:.2f}"
    return rows, summary_at_best


def plot_outputs(scan: pd.DataFrame, summary: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    ordered = summary.sort_values("hosting_capacity_mw")
    plt.bar(ordered["scenario_id"], ordered["hosting_capacity_mw"], color="#4c78a8")
    plt.ylabel("Hosting capacity (MW)")
    plt.xlabel("Scenario")
    plt.title("IEEE-33 demonstration hosting capacity by scenario")
    plt.xticks(rotation=30, ha="right")
    savefig(fig_dir / "18_hosting_capacity_by_scenario.png")

    plt.figure(figsize=(10, 5))
    sample = summary.sort_values("hosting_capacity_mw").head(4)["scenario_id"]
    for sid in sample:
        group = scan[scan["scenario_id"].eq(sid)].sort_values("dpv_capacity_mw")
        plt.plot(group["dpv_capacity_mw"], group["v_min"], marker="o", label=f"{sid} v_min")
        plt.plot(group["dpv_capacity_mw"], group["v_max"], linestyle="--", label=f"{sid} v_max")
    plt.axhline(V_LIMIT_LOW, color="red", linewidth=1)
    plt.axhline(V_LIMIT_HIGH, color="red", linewidth=1)
    plt.xlabel("DPV capacity (MW)")
    plt.ylabel("Voltage p.u.")
    plt.title("Voltage envelopes in limiting scenarios")
    plt.legend(ncol=2, fontsize=8)
    savefig(fig_dir / "19_voltage_envelope_examples.png")

    plt.figure(figsize=(9, 4.8))
    plt.bar(summary["scenario_id"], summary["reverse_power_hours"], color="#f58518")
    plt.ylabel("Reverse-power hours")
    plt.xlabel("Scenario")
    plt.title("Reverse-power screening indicator at hosting capacity")
    plt.xticks(rotation=30, ha="right")
    savefig(fig_dir / "20_reverse_power_hours_by_scenario.png")


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[OK] wrote {path}")


def write_report(output: Path, scan: pd.DataFrame, summary: pd.DataFrame, fig_dir: Path, baseline: dict[str, object], slack_voltage_pu: float) -> None:
    lines = [
        "# IEEE-33 hosting-capacity application demonstration",
        "",
        "## Network",
        "- Demonstration system: standard IEEE 33-node radial feeder.",
        f"- Topology: 33 nodes, {len(BRANCHES)} branches.",
        f"- Slack/substation voltage setpoint: {slack_voltage_pu:.3f} p.u.",
        f"- Baseline no-PV converged: {baseline['converged']}",
        f"- Baseline voltage range: {baseline['v_min']:.4f}-{baseline['v_max']:.4f} p.u.",
        "",
        "## Hosting-Capacity Summary",
        summary[
            [
                "scenario_id",
                "scenario_type",
                "cluster_id",
                "hosting_capacity_label",
                "hosting_capacity_mw",
                "v_min",
                "v_max",
                "reverse_power_hours",
                "reverse_energy_mwh",
                "max_branch_current_pu",
            ]
        ].to_markdown(index=False),
        "",
        "## Limitations",
        "- This is an IEEE-33 demonstration, not an actual western Hubei feeder.",
        "- No measured GHI, measured PV, or feeder SCADA validation is included.",
        "- Line current limits are reported as relative branch-current indicators.",
        "- No economic valuation, investment analysis, or storage-sizing claim is made.",
        "",
        "## Figures",
    ]
    lines.extend(f"- {p}" for p in sorted(fig_dir.glob("*.png")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="data/analysis/phase3/scenario_catalog.csv")
    parser.add_argument("--node-profiles", default="data/analysis/phase3/node_pv_profiles.csv")
    parser.add_argument("--output-dir", default="data/analysis/ieee33_application")
    parser.add_argument("--fig-dir", default="figures/ieee33_application")
    parser.add_argument("--report", default="reports/ieee33_application_demo.md")
    parser.add_argument("--capacity-max-mw", type=float, default=8.0)
    parser.add_argument("--capacity-step-mw", type=float, default=0.25)
    parser.add_argument("--slack-voltage-pu", type=float, default=1.05)
    args = parser.parse_args()

    scenarios = pd.read_csv(args.scenarios)
    profiles = pd.read_csv(args.node_profiles)
    capacities = np.round(np.arange(0.0, args.capacity_max_mw + args.capacity_step_mw / 2, args.capacity_step_mw), 6)
    baseline = run_power_flow({}, slack_voltage_pu=args.slack_voltage_pu)

    scan_rows = []
    summary_rows = []
    for _, scenario in scenarios.iterrows():
        rows, scenario_summary = evaluate_scenario(
            scenario,
            profiles,
            capacities,
            args.slack_voltage_pu,
        )
        scan_rows.extend(rows)
        summary_rows.append(scenario_summary)
        print(f"[OK] scenario={scenario['scenario_id']} hosting={scenario_summary['hosting_capacity_label']} MW")

    scan = pd.DataFrame(scan_rows)
    summary = pd.DataFrame(summary_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scan.to_csv(output_dir / "hosting_capacity_scan.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / "hosting_capacity_summary.csv", index=False, encoding="utf-8")
    print(f"[OK] wrote {output_dir / 'hosting_capacity_scan.csv'}")
    print(f"[OK] wrote {output_dir / 'hosting_capacity_summary.csv'}")

    fig_dir = Path(args.fig_dir)
    plot_outputs(scan, summary, fig_dir)
    write_report(Path(args.report), scan, summary, fig_dir, baseline, args.slack_voltage_pu)


if __name__ == "__main__":
    main()
