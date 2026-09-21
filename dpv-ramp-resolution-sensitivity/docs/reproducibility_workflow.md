# Reproducibility Workflow

Run commands from the repository root after installing `requirements.txt`.

## 1. Prepare Meteorological Data

Tianji data should be prepared locally according to provider access rules. The public code supports inspection, cropping, monthly merging, and point extraction:

```bash
python scripts/tj_preprocess.py inspect --file path/to/example.nc
python scripts/tj_preprocess.py merge \
  --input-root data/raw/tianji_authorized \
  --output-root data/processed/tianji_monthly \
  --start 2025-07-01 \
  --end 2025-07-31
```

ERA5 can be prepared either from an existing CDS NetCDF file or from the public ARCO Zarr store:

```bash
python scripts/era5_download_preprocess.py write-request \
  --month 2025-07 \
  --output data/raw/era5/era5_2025-07_request.json

python scripts/era5_download_preprocess.py download-arco \
  --month 2025-07 \
  --output-root data/processed/era5_monthly

python scripts/era5_download_preprocess.py map-to-tianji \
  --month 2025-07 \
  --era5-root data/processed/era5_monthly \
  --tianji-root data/processed/tianji_monthly \
  --output-root data/processed/era5_on_tianji_monthly
```

## 2. Convert Weather to PV

```bash
python scripts/pv_convert_grid.py \
  --input-root data/processed/tianji_monthly/2025-07 \
  --output data/processed/pv_grid/2025-07/pv_pu.nc

python scripts/pv_convert_grid.py \
  --input-root data/processed/era5_on_tianji_monthly/2025-07 \
  --output data/processed/era5_on_tianji_pv_grid/2025-07/pv_pu.nc
```

## 3. Detect and Cluster Ramps

```bash
python scripts/detect_ramps_grid.py \
  --input data/processed/pv_grid/2025-07/pv_pu.nc \
  --weather-root data/processed/tianji_monthly/2025-07 \
  --output data/processed/ramp_events/2025-07/ramp_events.csv \
  --thresholds 0.2 0.3 0.4

python scripts/cluster_ramps_grid.py \
  --input data/processed/ramp_events/2025-07/ramp_events.csv \
  --events-output data/processed/ramp_clusters/2025-07/ramp_events_clustered.csv \
  --clusters-output data/processed/ramp_clusters/2025-07/ramp_clusters.csv \
  --threshold 0.3
```

## 4. Run Resolution-Sensitivity Experiments

```bash
python scripts/run_ramp_resolution_experiments.py
python scripts/run_ghi_first_coarsening_experiment.py
```

This writes analysis tables to `data/analysis/seta_experiments/` and figure drafts to `figures/seta_experiments/`. These outputs are ignored by git.

`run_ramp_resolution_experiments.py` builds the baseline PV-field coarsening, ERA5 contrast, threshold, spatial-structure, monthly, NASA POWER, and IEEE33 summary tables. `run_ghi_first_coarsening_experiment.py` builds the weather-field-first coarsening tables by aggregating Tianji GHI, temperature, and wind before PV conversion. It also writes nearest-neighbor versus linear regridding sensitivity outputs.

## 5. Robustness and Benchmark Scripts

```bash
python scripts/run_pv_model_comparison.py
python scripts/run_stdbscan_parameter_scan.py
python scripts/run_nasa_power_station_benchmark.py \
  --station-metadata configs/station_metadata_template.csv
```

## 6. IEEE33 Application Demonstration

```bash
python scripts/build_phase3_scenarios_and_node_profiles.py
python scripts/run_ieee33_application_demo.py
```

The IEEE33 workflow is a technical application demonstration only. It is not calibrated feeder planning or economic valuation.
