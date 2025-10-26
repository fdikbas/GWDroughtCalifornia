# GWDroughtCalifornia
# Groundwater Drought Dynamics in California (2008–2021)
### Reproducible, observation-driven SGI/event/trend/persistence analysis from monthly well hydrographs

This repository hosts a single-file Python workflow and example input to reproduce all figures and tables reported in the manuscript:

**“Groundwater Drought Dynamics in California (2008–2021): Assessment with the Standardized Groundwater-Level Index, Event and Trend Metrics, and a Drought-Persistence Index.”**

The workflow ingests one monthly well dataset, builds a Standardized Groundwater-level Index (SGI), parses drought events, tests monotonic trends (Mann–Kendall with Sen’s slope), maps annual anomalies and network-scale persistence (GW-NDSPI), and exports publication-ready TIFFs (400 dpi) and CSVs—end-to-end and reproducibly.

---

## What’s here

- **`california_gw_all_in_one_v.1.0.py`**  
  Single-file, end-to-end analysis and figure engine. Parameters are declared near the top (CRS, IDW power, grid density, color scales, basemap behavior, figure sizes, etc.).  
  _Method summary and output directories embedded inside the file._ :contentReference[oaicite:1]{index=1}

- **`california_gw_minimal_long.csv`**  
  Example input (long/tidy format) with columns:
  - `Well` – well identifier (string)  
  - `LATITUDE`, `LONGITUDE` – WGS84 degrees (EPSG:4326)  
  - `Date` – timestamp (any parseable format; coerced to month start)  
  - `Value` – groundwater level WSE (ft a.m.s.l.)

---

## Key features

- **SGI** (unitless, per-site standardized departures) for cross-well comparability  
- **Event metrics** (NumEvents, MaxDroughtDuration, annual MinSGI, annual CumulativeDeficit) using SGI<−1  
- **Trends** via Mann–Kendall (τ, p) and **Sen’s slope** (per year)  
- **Annual anomaly maps** from WSE annual means relative to a 3-year per-well baseline  
- **GW-NDSPI** (negative-domain accumulation of sub-threshold SGI months) to condense multi-year persistence  
- **Mapping** with fast **IDW** (default) or **triangulated linear interpolation** (no extrapolation beyond the hull)  
- Publication-grade exports: **400-dpi TIFFs** and **machine-readable CSVs**

- Quick start

Put your input CSV (same schema as california_gw_minimal_long.csv) in the repo root.

In california_gw_all_in_one_v1.0.py, confirm:

CALIF_FILE = "california_gw_minimal_long.csv"


Run:

python california_gw_all_in_one_v1.0.py


Outputs appear under:

out_figs_EN/ (figures + aux_tables/)

out_SGI_spatial_EN/ (yearly SGI spatial metrics)

out_SGI_plots_EN/ (per-well SGI series)

out_annual_gw_maps_EN/ (absolute annual WSE)

out_annual_anomaly_maps_EN/ (annual WSE anomalies)

Each folder is described below. 


Note: Basemap tiles are optional and time-capped; everything runs without internet.

Input format

Long/tidy CSV with one record per (Well, Date), including coordinates:

Well	LATITUDE	LONGITUDE	Date	Value
21N…	37.123	−120.456	2008-01-15	1250

Dates are coerced to monthly (first-of-month) and then aligned to a monthly index.

Multiple readings within a month per well are averaged to the month.

Outputs (folders & files)

Tables (CSV) – out_figs_EN/aux_tables/

descriptive_stats_per_well.csv – per-well stats (mean, median, sd, quantiles, record length)

descriptive_stats_basin.csv – pooled (“basin-wide”) monthly WSE stats

study_area_bounds_lonlat.csv – lon/lat bounding box

sgi_drought_events.csv – full event log (start/end, duration, severity)

drought_metrics_yearly.csv – annual per-well metrics (NumEvents, MaxDroughtDuration, MinSGI, CumulativeDeficit)
All paths produced by the script. 


Figures (TIFF, 400 dpi) – out_figs_EN/

fig0_study_area_map.tiff / fig0b_california_locator_map.tiff

fig2_annual_mean_heatmap.tiff (annual mean SGI; wells × years)

fig2_annual_mean_heatmap_WSE.tiff (annual WSE anomalies; wells × years)

fig3_sgi_sen_slope_per_well.tiff (per-well Sen’s slope, SGI per year; * marks p<0.05)
(Size & palette match the manuscript figures; color semantics consistent.) 


Spatial drought-metric rasters (TIFF) – out_SGI_spatial_EN/

For each year:
MaxDroughtDuration_<YYYY>.tiff, MinSGI_<YYYY>.tiff, CumulativeDeficit_<YYYY>.tiff, NumEvents_<YYYY>.tiff
(IDW on a fixed extent; consistent color logic “red=worse” for all metrics.) 


Per-well SGI series (TIFF) – out_SGI_plots_EN/

SGI_series_<WELL>.tiff with shaded severity bands and monthly markers. 

Annual WSE maps (TIFF) – out_annual_gw_maps_EN/

Interpolated absolute annual WSE (IDW), fixed extent, consistent DPI. 

Annual WSE anomaly maps (TIFF) – out_annual_anomaly_maps_EN/

Diverging, zero-centered scale across all years for direct comparability. 

Configuration knobs (in the script)

Interpolation method: INTERP_METHOD = "idw" or "tri" (triangulated linear = no extrapolation beyond hull)

IDW power: IDW_POWER (e.g., 1.1–2.0; lower = smoother, higher = locally sharper)

Grid density: IDW_NX_DEFAULT, IDW_MAX_PIXELS, TRI_NX_DEFAULT (controls crispness vs. speed)

Figure sizes & DPI: FIG_W, FIG_H, savefig.dpi=400

Anomaly baseline: first three complete years per well (configurable)

Color scales: fixed or robust quantile-based (2–98%) with symmetric handling for anomalies

Basemaps: optional (time-capped); tiles skipped automatically if slow/unavailable

CRS: EPSG:4326 for input; mapping in California Albers (EPSG:3310) 


Reproducibility

One script + one CSV → all outputs.

No manual clicks; all parameters are versioned and declared at the top of the script.

Figures are exported at fixed size/DPI to avoid layout drift.

Color scales are either fixed or data-aware but consistent across years. 


CITE:

If this code or outputs support your work, please cite the article (when available) and this repository (see CITATION.cff). A short software citation is:

Dikbaş, F. (2025). GWDroughtCalifornia — Reproducible SGI/event/trend/persistence workflow (v1.0). GitHub: https://github.com/fdikbas/GWDroughtCalifornia

License

This project is released under GNU General Public License v3.0 (GPL-3.0).

GPL-3.0 ensures that derivative works remain open (copyleft), which can be desirable for public-interest water resources tools.

If you prefer broader reuse in proprietary settings, consider Apache-2.0 (permissive, explicit patent grant).

See LICENSE.

Contributing

Issues and pull requests are welcome (bug reports, data-format clarifications, performance improvements, new figure templates). See CONTRIBUTING.md.

Troubleshooting

Basemaps slow or blocked? Set SKIP_BASEMAPS = True. All analytics/figures still build.

Memory on large grids? Lower IDW_NX_DEFAULT or IDW_MAX_PIXELS.

Interpolation artifacts (islands/gaps)? Try INTERP_METHOD="tri" for piecewise-linear fields with NaN outside the well hull; or lower IDW_POWER (e.g., 1.1) for smoother fields.

Pixelation in maps? Increase IDW_NX_DEFAULT or TRI_NX_DEFAULT, and keep savefig.dpi=400.

Module not found: Install via requirements.txt or environment.yml.


---

If you don’t need basemaps, you can omit contextily, geopandas, and shapely.
- No manual steps—results regenerate from the single script + CSV. :contentReference[oaicite:2]{index=2}

