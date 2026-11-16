# Data access and sensitivity boundary

The repository publishes code and selected aggregate summaries. It does not
publish the purchased station observations or the gridded source products used
to generate them.

## Excluded from GitHub

- station observation files and station-level metrics;
- Tianji or ERA5 NetCDF/GRIB/Zarr fields and processed grids;
- cached NASA POWER responses;
- Himawari collections and event-level cloud files;
- terrain tiles, local administrative-boundary caches, and private metadata;
- credentials, download logs containing account information, and local paths.

The root `.gitignore` blocks common raw-data and credential extensions. The
ignore file is a guardrail, not a substitute for reviewing a commit before
pushing it.

## Included summaries

`seta_revision/data/aggregate/` contains compact product- or experiment-level
tables. They do not contain station IDs, coordinates, raw hourly values, or
event timestamps. Remove an aggregate table if its provider agreement does not
permit redistribution of derived summaries.

## How users obtain inputs

Users must obtain each source product from its official provider or from the
authors under the applicable purchase or access terms. Record the product
identifier, version, retrieval date, units, time convention, and required
attribution in the local project. Set `SETA_PROJECT_ROOT` to that local project
when running the scripts.

The MIT notice in this repository covers the code. It does not change the
license or redistribution terms of any purchased or third-party data product.
