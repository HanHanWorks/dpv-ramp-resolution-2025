# Station-referenced photovoltaic ramp analysis

Code and selected aggregate results accompanying the SETA manuscript
"Spatial support and product physics jointly control station-referenced
photovoltaic ramp characterization in complex terrain".

The current revision package is under [`seta_revision/`](seta_revision/).
The older [`dpv-ramp-resolution-sensitivity/`](dpv-ramp-resolution-sensitivity/)
directory is retained as a legacy workflow snapshot and is not the source of
the current revision results.

## What is public

This repository contains analysis code, configuration-independent scripts,
dependency lists, documentation, and a small set of aggregate summary tables.
The aggregate tables contain no station-level identifiers or raw observations.

Purchased or provider-controlled source data remain private and are not stored
in GitHub. This includes station observations, Tianji and ERA5 fields, cached
NASA POWER responses, Himawari collections, terrain caches, NetCDF/HDF5 files,
and detailed event or timestamp tables. Obtain those inputs independently and
under the applicable provider terms.

The software license applies to the code only. It does not grant permission to
redistribute source data or impose a license on third-party data products.

## Current revision layout

```text
seta_revision/
  01_revision_package/       table-rebuild helper scripts
  02_evidence_and_analysis/  current station-reference analysis scripts
  04_build_and_qa/           figure-rebuild helper
  data/aggregate/             selected non-station summary tables
  docs/                       data boundary and reproduction notes
  requirements.txt
```

## Reproduction boundary

The scripts that rebuild results require an authorized local project containing
the purchased/private inputs. Set `SETA_PROJECT_ROOT` to that local project;
the scripts do not download or publish those files. The included aggregate
tables allow inspection of selected published summaries without access to the
source observations.

For a raw-data rerun, install the revision dependencies and pass the private
input paths explicitly:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r seta_revision/requirements.txt
export SETA_PROJECT_ROOT=/absolute/path/to/the/private/source/project
python seta_revision/02_evidence_and_analysis/analyses/run_seta_canonical_station_scan_consecutive_hours.py \
  --stations "$SETA_PROJECT_ROOT/paper_writing_package_seta_major_revision/01_data/station_reference/integrated_15site_validation_hourly_repaired.csv" \
  --data-root "$SETA_PROJECT_ROOT/data" \
  --output-dir "$SETA_PROJECT_ROOT/data/analysis/seta_revision" \
  --months 2025-04 2025-07 2025-10 \
  --thresholds 100 150 200 250 300 350 400 \
  --tolerances 0 1 2 3
```

The raw-data workflow is intentionally not claimed to be self-contained in
this repository. See [`seta_revision/docs/data_access_and_sensitivity.md`](seta_revision/docs/data_access_and_sensitivity.md)
and [`seta_revision/docs/reproducibility_workflow.md`](seta_revision/docs/reproducibility_workflow.md).

## Release policy

The GitHub repository is the code and small-summary-data record. Release
`V1.0.1` is archived on Zenodo at
[`10.5281/zenodo.24273521`](https://doi.org/10.5281/zenodo.24273521). Future
deposits should use a reviewed tag from this repository and must not include
the private source datasets. Do not put credentials, local absolute paths,
station-level files, or raw provider downloads into an issue, release asset,
or commit.
