# GWDroughtCalifornia
# Groundwater Drought Dynamics in California (2008–2021)
### Reproducible, observation-driven SGI/event/trend/persistence analysis from monthly well hydrographs

This repository hosts a single-file Python workflow and example input to reproduce all figures and tables reported in the manuscript:

**“Groundwater Drought Dynamics in California (2008–2021): Assessment with the Standardized Groundwater-Level Index, Event and Trend Metrics, and a Drought-Persistence Index.”**

The workflow ingests one monthly well dataset, builds a Standardized Groundwater-level Index (SGI), parses drought events, tests monotonic trends (Mann–Kendall with Sen’s slope), maps annual anomalies and network-scale persistence (GW-NDSPI), and exports publication-ready TIFFs (400 dpi) and CSVs—end-to-end and reproducibly.

---

## What’s here

- **`california_gw_all_in_one_EN_YYYY.MM.DD.v*.py`**  
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
- No manual steps—results regenerate from the single script + CSV. :contentReference[oaicite:2]{index=2}

