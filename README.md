# DPV Ramp Resolution Sensitivity in Complex Terrain

Public algorithm code supporting the manuscript *"Spatial support and product  
physics jointly control station-referenced photovoltaic ramp characterization in  
complex terrain"*, submitted to *Sustainable Energy Technologies and  
Assessments* (SETA, Elsevier).

## Scope

The code supports:

- gridded meteorological preprocessing and time alignment;
- weather-to-PV conversion using simplified, reproducible PV models;
- gridded PV ramp-event detection;
- ST-DBSCAN ramp-cluster identification;
- controlled weather-field-first and PV-field-first spatial coarsening experiments;
- terrain-conditioned ramp summaries;
- PV-model, ramp-threshold, and clustering-parameter sensitivity checks;
- NASA POWER open-product benchmarking;
- IEEE33 hosting-capacity application demonstration.

The repository intentionally does not include Tianji raw data, ERA5 extracts, NASA POWER cached responses, terrain tile caches, processed NetCDF files, manuscript figures, or analysis result tables.

## Repository Layout

```text
configs/
  monthly_experiment_plan.csv
  station_metadata_template.csv
docs/
  data_access_and_sensitivity.md
  reproducibility_workflow.md
scripts/
  *.py
README.md
requirements.txt
.gitignore
LICENSE
CITATION.cff
```

## Data Boundary

No sensitive or restricted data are included. To reproduce the manuscript analyses, place locally authorized input data under the ignored `data/` directory following the relative paths described in `docs/reproducibility_workflow.md`.

NASA POWER is an open product and can be downloaded by `scripts/run_nasa_power_station_benchmark.py`. Tianji and some local processed products may be subject to access restrictions; do not upload them unless you have redistribution rights.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_ramp_resolution_experiments.py
python scripts/run_ghi_first_coarsening_experiment.py
```

The quick-start command assumes the required local input files already exist under `data/processed/`. See `docs/reproducibility_workflow.md` for the expected pipeline order and paths.

## Claim Boundary

This code is for resolution-sensitivity analysis and reproducible method reporting. It does not provide measured GHI/PV validation, plant-level PV calibration, real-feeder planning, economic valuation, investment analysis, or storage-sizing optimization.
