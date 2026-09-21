# Data Access and Sensitivity Boundary

This public code package excludes all restricted, sensitive, or bulky data.

## Not Included

- Tianji raw or processed NetCDF files;
- ERA5 regional NetCDF extracts;
- local station metadata beyond the placeholder template;
- NASA POWER cached JSON/CSV responses;
- terrain tile caches;
- generated ramp-event CSVs, cluster CSVs, analysis tables, and figures;
- manuscript drafts or private review notes.

## Safe to Include

- algorithm scripts;
- non-sensitive configuration templates;
- documentation describing the expected directory layout;
- manuscript-independent code metadata.

## Expected Local Data Layout

```text
data/
  processed/
    tianji_monthly/YYYY-MM/
      DSWRFsfc.nc
      t2mz.nc
      UGRD10m.nc
      VGRD10m.nc
      u100m.nc
      v100m.nc
    era5_monthly/YYYY-MM/
      DSWRFsfc.nc
      t2mz.nc
      UGRD10m.nc
      VGRD10m.nc
      u100m.nc
      v100m.nc
    era5_on_tianji_monthly/YYYY-MM/
      DSWRFsfc.nc
      t2mz.nc
      UGRD10m.nc
      VGRD10m.nc
      u100m.nc
      v100m.nc
    pv_grid/YYYY-MM/
      pv_pu.nc
    era5_on_tianji_pv_grid/YYYY-MM/
      pv_pu.nc
    terrain/
      terrain_classes.csv
```

Only upload data if the license and provider terms explicitly allow redistribution.

