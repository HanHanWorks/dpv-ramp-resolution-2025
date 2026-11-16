#!/usr/bin/env python3
"""Reconcile event-count pooled and station-month macro frequency bias."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
CANONICAL = HERE / "consecutive_hour_primary_rerun" / "canonical_main_results.csv"
FAIR_MASK = HERE / "fair_mask_common_month_product_summary.csv"
OUT = HERE / "frequency_bias_estimand_reconciliation.csv"

rows = []
canonical = pd.read_csv(CANONICAL)
canonical = canonical.query("threshold_wm2h == 200 and tolerance_h == 1 and solar_qc_group == 'primary_solar_qc'")
for _, r in canonical.iterrows():
    rows.append({
        "estimand_scope": "canonical_product_specific_mask",
        "months": "2025-04,2025-07,2025-10" if r["source"] == "tianji" else "2025-04,2025-07",
        "product": r["product"],
        "stations": int(r["stations"]),
        "station_months": int(r["station_months"]),
        "macro_station_month_frequency_bias": float(r["frequency_bias_mean"]),
        "observed_events_sum": int(r["obs_events_sum"]),
        "modelled_events_sum": int(r["model_events_sum"]),
        "pooled_event_count_frequency_bias": float(r["model_events_sum"] / r["obs_events_sum"]),
    })

fair = pd.read_csv(FAIR_MASK).query("tolerance_h == 1")
for _, r in fair.iterrows():
    rows.append({
        "estimand_scope": "all_products_common_hour_mask",
        "months": "2025-04,2025-07",
        "product": r["product"],
        "stations": int(r["stations"]),
        "station_months": int(r["station_months"]),
        "macro_station_month_frequency_bias": float(r["macro_frequency_bias"]),
        "observed_events_sum": int(r["observed_events_sum"]),
        "modelled_events_sum": int(r["model_events_sum"]),
        "pooled_event_count_frequency_bias": float(r["model_events_sum"] / r["observed_events_sum"]),
    })

pd.DataFrame(rows).sort_values(["estimand_scope", "product"]).to_csv(OUT, index=False, float_format="%.8g")
print(pd.read_csv(OUT).to_string(index=False))
