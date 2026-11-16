# Reproducibility workflow

Run from the repository root after installing
`seta_revision/requirements.txt`.

## 1. Prepare a private source project

Keep the source project outside this Git checkout. It must contain the
authorized station table and processed meteorological products under the
relative layout expected by the scripts. Set:

```bash
export SETA_PROJECT_ROOT=/absolute/path/to/the/private/source/project
```

Do not copy that directory into the repository and do not commit its contents.

## 2. Rebuild the canonical station scan

```bash
python seta_revision/02_evidence_and_analysis/analyses/run_seta_canonical_station_scan_consecutive_hours.py \
  --stations "$SETA_PROJECT_ROOT/paper_writing_package_seta_major_revision/01_data/station_reference/integrated_15site_validation_hourly_repaired.csv" \
  --data-root "$SETA_PROJECT_ROOT/data" \
  --output-dir "$SETA_PROJECT_ROOT/data/analysis/seta_revision" \
  --months 2025-04 2025-07 2025-10 \
  --thresholds 100 150 200 250 300 350 400 \
  --tolerances 0 1 2 3
```

The generated detailed outputs stay in the private source project. Only
reviewed aggregate summaries may be copied into `data/aggregate/`.

## 3. Rebuild derived summaries

The table and figure helper scripts use the retained analysis layout and
require the detailed private outputs. Run them against a private working copy,
then review every output before publication. The public repository does not
promise that these helpers run without the restricted inputs.

## 4. Pre-push checks

From the repository root:

```bash
python -m compileall -q seta_revision
git diff --check
git status --short
```

Run a local secret and absolute-path scanner over the staged files before each
push. Review `git diff --cached --stat` and the complete staged file list as
well. A passing syntax check does not establish scientific validity or data
redistribution permission.
