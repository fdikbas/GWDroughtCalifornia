"""
California GW Drought — ALL-IN-ONE analysis (EN) — complete, stable, map-ready

Inputs
  • california_gw_minimal_long.csv  (columns: Well, LATITUDE, LONGITUDE, Date, Value[WSE ft a.m.s.l.])

Outputs (English labels; _EN folders)
  • Tables (CSV): out_figs_EN/aux_tables/*
  • Publication-ready TIFFs (400 dpi): out_figs_EN/*.tiff
  • Spatial maps of yearly drought metrics (SGI): out_SGI_spatial_EN/*.tiff
  • SGI monthly time-series for each well: out_SGI_plots_EN/SGI_series_<WELL>.tiff
  • Interpolated annual groundwater level maps (IDW): out_annual_gw_maps_EN/*.tiff
  • Interpolated annual groundwater anomaly maps (IDW): out_annual_anomaly_maps_EN/*.tiff
"""

# SPDX-License-Identifier: GPL-3.0-only
#
# Groundwater Drought Dynamics in California — SGI/Event/Trend/GW-NDSPI Workflow
# Copyright (c) 2025 Fatih Dikbaş
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# Project repo: https://github.com/fdikbas/GWDroughtCalifornia
# Suggested citation (software):
#   Dikbaş, F. (2025). GWDroughtCalifornia — Reproducible SGI/event/trend/
#   persistence workflow (v1.0). GitHub. https://github.com/fdikbas/GWDroughtCalifornia
#
# Notes:
# - Input: long-format monthly groundwater levels (Well, LATITUDE, LONGITUDE,
#   Date, Value). Coordinates: EPSG:4326.
# - Outputs: 400-dpi TIFF figures and CSV tables (events, trends, anomalies,
#   GW-NDSPI), plus per-well diagnostics.
# - Optional basemaps (if enabled) remain subject to their respective provider
#   terms; this script runs without internet if basemaps are disabled.

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import colors as mcolors
# --- Triangulated linear interpolation (no extrapolation beyond hull) ---
from matplotlib.tri import Triangulation, LinearTriInterpolator
import contextily as cx

from scipy.stats import kendalltau
# ---------- GLOBAL HELPERS (make available to all figures) ----------
# ---------- GLOBAL HELPERS (available everywhere) ----------
# --- Natural sort + seyrek y-etiket yardımcıları: TEK TANIM (global) ---
import re
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', str(s))]

def _sorted_well_order(well_list):
    return sorted(list(well_list), key=_natural_key)

def _sparse_y_labels(labels, max_labels=24):
    n = len(labels)
    if n <= max_labels:
        idx = list(range(n))
    else:
        idx = list(np.linspace(0, n-1, max_labels).round().astype(int))
    return idx, [str(labels[i]) for i in idx]

def _add_basemap_safe(ax, source, crs, zoom=None, **kwargs):
    if zoom is None:
        return cx.add_basemap(ax, source=source, crs=crs, **kwargs)
    else:
        return cx.add_basemap(ax, source=source, crs=crs, zoom=int(zoom), **kwargs)

def add_basemap_with_big_labels(ax, base="Esri.WorldTopoMap", base_zoom=10,
                                label_zoom=12, crs="EPSG:3857"):
    """
    Fast/robust basemap adder:
      - respects SKIP_BASEMAPS
      - avoids label overlay unless BASE_TRY_LABELS=True
      - returns quickly if tile fetch seems slow
    """
    if 'SKIP_BASEMAPS' in globals() and SKIP_BASEMAPS:
        return False  # skip tiles entirely

    import time
    prov = cx.providers
    for part in base.split("."):
        prov = getattr(prov, part)

    t0 = time.time()
    try:
        cx.add_basemap(ax, source=prov, crs=crs, attribution=False, zoom=base_zoom)
    except Exception as e:
        print("[WARN] Base tiles failed fast:", e)
        return False

    if (time.time() - t0) * 1000.0 > BASE_FETCH_MS:
        # Too slow: do not try labels
        return True

    if ('BASE_TRY_LABELS' in globals() and BASE_TRY_LABELS):
        try:
            label_src = cx.providers.CartoDB.PositronOnlyLabels
            cx.add_basemap(ax, source=label_src, crs=crs, attribution=False,
                           zoom=label_zoom, alpha=1.0)
        except Exception as e:
            print("[WARN] Label overlay failed:", e)
    return True

# ---- optional geo stack (kept optional to never hang) ----
try:
    import geopandas as gpd
    from shapely.geometry import Point, box
    HAS_GEO = True
except Exception:
    HAS_GEO = False

# ---- progress bars ----
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        if iterable is None:
            class _Dummy:
                def __enter__(self): return self
                def __exit__(self, *exc): pass
                def update(self, *a, **k): pass
                def close(self): pass
            return _Dummy()
        return iterable

def add_scalebar(ax,
                 length_km: float = 50.0,
                 location: str | None = None,   # new name
                 loc: str | None = None,        # backward compatibility
                 pad_frac: float = 0.025,
                 height_frac: float = 0.012,
                 facecolor: str = "white",
                 edgecolor: str = "black",
                 text_color: str = "black",
                 lw: float = 0.9,
                 fontsize: int = 9):
    """
    Draw a simple metric scalebar on a map in EPSG:3857 (meters).
    Accepts both 'location=' and legacy 'loc='.
    Valid positions contain 'lower'/'upper' and 'left'/'right' (or 'center').
    """
    # resolve parameter naming
    if location is None and loc is not None:
        location = loc
    if location is None:
        location = "lower left"

    # current data extent (must be set *after* setting basemap/limits)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= 0 or dy <= 0:
        # nothing to draw against
        return

    # geometry in data units (EPSG:3857 meters)
    bar_len_m = float(length_km) * 1000.0
    bar_h = height_frac * dy
    pad_x = pad_frac * dx
    pad_y = pad_frac * dy

    # vertical placement
    if "lower" in location:
        y0 = ymin + pad_y
    else:
        y0 = ymax - pad_y - bar_h

    # horizontal placement
    if "left" in location:
        x0 = xmin + pad_x
    elif "right" in location:
        x0 = xmax - pad_x - bar_len_m
    else:  # center
        x0 = xmin + 0.5 * (dx - bar_len_m)

    # draw bar
    rect = patches.Rectangle((x0, y0), bar_len_m, bar_h,
                             facecolor=facecolor, edgecolor=edgecolor, lw=lw,
                             zorder=10)
    ax.add_patch(rect)

    # label (km)
    ax.text(x0 + bar_len_m / 2.0, y0 + bar_h * 1.2,
            f"{int(length_km)} km",
            ha="center", va="bottom",
            color=text_color, fontsize=fontsize,
            zorder=11)

def add_north_arrow(ax,
                    xy=(0.92, 0.14),
                    length=0.08,
                    color="black",
                    lw=1.2,
                    head_length=12,
                    fontsize=10,
                    loc=None):
    """
    North arrow in axes-fraction coordinates; independent of CRS.
    Accepts either 'xy=(x,y)' in axes fraction OR a convenience 'loc' like
    'upper left', 'upper right', 'lower left', 'lower right'.
    """
    if isinstance(loc, str):
        loc_map = {
            "upper left":  (0.12, 0.82),
            "upper right": (0.88, 0.82),
            "lower left":  (0.12, 0.12),
            "lower right": (0.88, 0.12),
        }
        xy = loc_map.get(loc.lower().strip(), xy)

    x, y = xy
    ax.annotate("",
                xy=(x, y + length), xytext=(x, y),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0,
                                mutation_scale=head_length),
                zorder=12)
    ax.text(x, y + length + 0.02, "N",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=fontsize, color=color, zorder=13)

# ---------------- PATHS & CONFIG ----------------
CALIF_FILE = "california_gw_minimal_long.csv"   # Well, LATITUDE, LONGITUDE, Date, Value (WSE ft)

OUT_FIGS       = Path("./out_figs_EN")
OUT_TBLS       = OUT_FIGS / "aux_tables"
OUT_SPATIAL    = Path("./out_SGI_spatial_EN")
OUT_SGI_SERIES = Path("./out_SGI_plots_EN")
OUT_ANNUAL     = Path("./out_annual_gw_maps_EN")
OUT_ANOMALY    = Path("./out_annual_anomaly_maps_EN")

for p in [OUT_FIGS, OUT_TBLS, OUT_SPATIAL, OUT_SGI_SERIES, OUT_ANNUAL, OUT_ANOMALY]:
    p.mkdir(parents=True, exist_ok=True)

# ---- style & switches ----
AUTO_TIME_FROM_DATA    = True
SHOW_MAP_LABELS        = False                 # NEVER show well names on maps
LOCATOR_PAD_FRAC       = 0.10
BASEMAP_PRIORITY       = ["Esri.WorldImagery", "Esri.WorldTopoMap", "CartoDB.Voyager", "OpenStreetMap.Mapnik"]

ROLL12_COL = "yellowgreen"
ROLL36_COL = "plum"
ROLL_ALPHA_12 = 0.95
ROLL_ALPHA_36 = 0.95
ROLL_LW_12 = 1.9
ROLL_LW_36 = 2.1

# ---- light-weight map settings ----
# Keep tiles crisp enough without going high-zoom or using bounds2img
MAX_Z_STUDY    = 12   # study-area tiles (previously tried 14)
MAX_Z_LOCATOR  = 8    # locator tiles
USE_HR_LOCATOR = False  # skip the HR locator map entirely

# Softer IDW defaults to prevent large arrays
IDW_NX_DEFAULT       = 448     # was 320–300 in places
IDW_PREFER_CELL_M    = 200.0   # ~0.2 km cells by default
IDW_MAX_PIXELS       = 80_000  # hard cap on grid size
IDW_CHUNK_ROWS       = 192     # small row tiles for memory friendliness

# Interpolation method selector for spatial maps
INTERP_METHOD   = "idw"   # "tri" for triangulated linear, "idw" to keep current IDW
TRI_NX_DEFAULT  = 600     # grid size for TRI (square); raise for crisper rasters
IDW_NX_DEFAULT  = 448     # keep your existing values
IDW_POWER       = 1.1

# Locator extent (lon_min, lon_max, lat_min, lat_max) — Western US view
LOCATOR_BOUNDS_LONLAT = (-128.0, -112.0, 31.0, 45.0)  # adjust if you want tighter/wider
REGION_LABEL = "California"
USE_HILLSHADE = False  # True yaparsanız deneyecek, yoksa pas geçer

# ---------- FAST START / BASEMAP CONTROLS ----------
DEFER_BASEMAPS   = True   # True = build all analytics & figures first, basemaps last
SKIP_BASEMAPS    = False   # True = never fetch tiles (scatter-only fallback); set False to enable tiles
BASE_TRY_LABELS  = False  # False = don't overlay label tiles (saves time & extra HTTP calls)
BASE_FETCH_MS    = 1200   # pseudo-timeout for basemap step (ms): return quickly if slow

# WSE variation focus
USE_WSE_ANOM_FOR_ANNUAL_MAPS = False
BASELINE_YEARS = [2008, 2009, 2010]
ANOM_CMAP     = "RdBu"                         # negative (decline) = red; positive = blue
SGI_CMAP      = "RdBu"                         # negative SGI (drought) = red

# --- WELL ID handling ---
NORMALIZE_WELL_IDS = False  # <- use raw well names exactly as in the CSV

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 400,
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.figsize": (6.0, 3.8),
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- Paper-wide figure size (match Figure 4) ---
FIG_W, FIG_H = 9.8, 9.8   # inches; set what you want Figure 4 to be

# ---- Joblib caching control ----
from joblib import Memory
ENABLE_CACHE = False   # turn OFF to avoid hashing/serializing big args
CACHE_DIR    = "./_joblib_cache"

MEM = Memory(location=CACHE_DIR if ENABLE_CACHE else None, verbose=0)

def save_tiff(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, format="tiff", dpi=400, bbox_inches="tight")
    plt.close()

# ---------------- HELPERS ----------------
def _norm_well_id(x) -> str:
    if pd.isna(x): return ""
    s = str(x).strip().replace("\u200b", "")
    s2 = s.replace(",", ".")
    try:
        f = float(s2)
        if np.isfinite(f) and abs(f - round(f)) < 1e-12:
            return str(int(round(f)))
        return s
    except Exception:
        return s

def _resolve_cx_provider(name: str):
    prov = cx.providers
    for part in str(name).split("."):
        prov = getattr(prov, part)
    return prov

def load_california_long(path: str):
    """
    Input: long/tidy CA file columns:
      Well, LATITUDE, LONGITUDE, Date, Value (WSE in ft a.m.s.l.)
    Returns: (df_wide_monthly, coords_df)
    """
    ca = pd.read_csv(path, encoding="utf-8-sig")
    req = {"Well","LATITUDE","LONGITUDE","Date","Value"}
    missing = req - set(ca.columns)
    if missing:
        raise ValueError(f"California file is missing columns: {missing}")
    ca["Date"] = pd.to_datetime(ca["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    ca = ca.dropna(subset=["Date","Value","Well"])

    # wide monthly table
    ts_df = (ca.pivot_table(index="Date", columns="Well", values="Value", aggfunc="mean")
               .sort_index().asfreq("MS"))

    # coordinates (one row per well label as-is)
    coords_df = (ca[["Well","LONGITUDE","LATITUDE"]]
                 .dropna()
                 .drop_duplicates(subset=["Well"])   # keep 1 coordinate per well label
                 .rename(columns={"LONGITUDE":"X","LATITUDE":"Y"}))
    coords_df["GroundElev"] = np.nan

    # DO NOT normalize/modify well IDs
    if NORMALIZE_WELL_IDS:
        # (leave as no-op by default)
        pass

    return ts_df, coords_df

def calculate_sgi(ts_df: pd.DataFrame) -> pd.DataFrame:
    return (ts_df - ts_df.mean()) / ts_df.std(ddof=0)

def extract_drought_events(series: pd.Series, thr=-1.0):
    events, in_evt = [], False
    start, sev, dur = None, 0.0, 0
    for date, val in series.items():
        if pd.notna(val) and val < thr:
            if not in_evt:
                in_evt = True; start = date; sev = 0.0; dur = 0
            sev += abs(val); dur += 1
        else:
            if in_evt:
                events.append({"well": series.name, "start": start,
                               "end": date - pd.offsets.MonthBegin(1),
                               "duration_mo": dur, "severity_sgi": sev})
                in_evt = False
    if in_evt:
        events.append({"well": series.name, "start": start,
                       "end": series.index[-1], "duration_mo": dur, "severity_sgi": sev})
    return events

def drought_metrics_yearly(sgi_df: pd.DataFrame, threshold=-1.0) -> pd.DataFrame:
    rows = []
    for well in tqdm(sgi_df.columns, desc="Yearly SGI metrics (per well)"):
        s = sgi_df[well].dropna()
        if s.empty: continue
        for year in sorted(set(s.index.year)):
            s_year = s[s.index.year == year]
            mask = s_year < threshold
            max_dur, curr, ev_count = 0, 0, 0
            for flag in mask:
                if flag: curr += 1
                else:
                    if curr > 0: max_dur = max(max_dur, curr); ev_count += 1; curr = 0
            if curr > 0: max_dur = max(max_dur, curr); ev_count += 1
            min_sgi = s_year[mask].min() if mask.any() else np.nan
            cum_def = float((-s_year[mask]).sum()) if mask.any() else 0.0
            rows.append({
                "Well": str(well),          # raw label
                "Year": int(year),
                "MaxDroughtDuration": int(max_dur),
                "MinSGI": float(min_sgi) if pd.notna(min_sgi) else np.nan,
                "CumulativeDeficit": cum_def,
                "NumEvents": int(ev_count)
            })
    return pd.DataFrame(rows)

def mann_kendall_sen(series_monthly: pd.Series) -> Tuple[float,float,float]:
    s = series_monthly.dropna()
    if len(s) < 10: return np.nan, np.nan, np.nan
    x = np.arange(len(s))
    tau, p = kendalltau(x, s.values)
    n = len(s); vals = s.values
    slopes = []
    for i in range(n - 1):
        dv = (vals[i + 1:] - vals[i]) / (np.arange(i + 1, n) - i)
        slopes.extend(dv.tolist())
    sen_month = np.median(slopes)
    return float(tau), float(p), float(sen_month * 12.0)

# ================== MEMORY-SAFE IDW GRID ==================
def _idw_grid(x, y, z,
              gx0, gx1, gy0, gy1,
              nx=None, ny=None,
              power=2.0,
              prefer_cell_m=5000.0,
              max_pixels=120_000,
              chunk_rows=256,
              dtype=np.float32,
              **kw):  # <— accept stray keywords
    # --- accept legacy keyword names for bounds ---
    if "xmin" in kw: gx0 = float(kw.pop("xmin"))
    if "xmax" in kw: gx1 = float(kw.pop("xmax"))
    if "ymin" in kw: gy0 = float(kw.pop("ymin"))
    if "ymax" in kw: gy1 = float(kw.pop("ymax"))
    # (optional) ignore unknown leftovers:
    kw.clear()
    
    """
    Bellek-dostu IDW ızgarası.
    - Büyük 3B yayın (broadcast) dizileri **oluşturmaz**.
    - (ny, nx) bloklarını satır-satır işleyip akümülatörle toplar.
    - Girdi koordinatlarının **metre** cinsinden olduğu varsayılır (EPSG:3310 vs).

    Dönüş:
        Xi (1D), Yi (1D), Zi (ny x nx), extent=(gx0,gx1,gy0,gy1)
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    # ---- Izgara çözünürlüğünü akıllı seç ----
    if nx is None or ny is None:
        W = float(gx1 - gx0)
        H = float(gy1 - gy0)

        if prefer_cell_m and prefer_cell_m > 0.0:
            nx_est = int(np.ceil(W / prefer_cell_m))
            ny_est = int(np.ceil(H / prefer_cell_m))
        else:
            nx_est, ny_est = 320, 240

        nx_est = max(80, min(1000, nx_est))
        ny_est = max(80, min(1000, ny_est))

        # Toplam pikseli sınırlayalım
        tot = nx_est * ny_est
        if tot > max_pixels:
            scale = (tot / max_pixels) ** 0.5
            nx = max(80, int(nx_est / scale))
            ny = max(80, int(ny_est / scale))
        else:
            nx, ny = nx_est, ny_est
    else:
        nx = int(nx); ny = int(ny)

    xs = np.linspace(gx0, gx1, nx, dtype=np.float64)
    ys = np.linspace(gy0, gy1, ny, dtype=np.float64)

    Zi = np.empty((ny, nx), dtype=dtype)

    # ---- Satır-karosu döngüsü ----
    # Akümülatörler her karoda yeniden oluşturulacak
    p = float(power)
    eps = 1e-12

    # Karoda kullanılacak satır yüksekliğini, nx'e göre güvenli seç
    # (çok geniş nx ise karoyu küçültüyoruz)
    if nx > 600:
        chunk_rows = min(chunk_rows, 192)
    if nx > 800:
        chunk_rows = min(chunk_rows, 128)

    j0 = 0
    while j0 < ny:
        j1 = min(ny, j0 + chunk_rows)
        yy = ys[j0:j1][:, None]        # (m, 1)
        # xs: (nx,) —> (1, nx)
        xx = xs[None, :]                # (1, nx)

        # Akümülatörler
        num = np.zeros((j1 - j0, nx), dtype=np.float64)
        den = np.zeros((j1 - j0, nx), dtype=np.float64)

        # Noktalar üzerinden tek tek geç (198 ~ küçük; bellek güvenli)
        for i in range(x.shape[0]):
            dx = xx - x[i]              # (1, nx)
            dy = yy - y[i]              # (m, 1)
            # (m, nx): yayın ile 2D uzaklık
            dist2 = dx * dx + dy * dy

            # Hücre noktanın tam üstünde ise z değerini birebir ata
            zero_mask = dist2 < eps
            if zero_mask.any():
                num[zero_mask] = z[i]
                den[zero_mask] = 1.0
                # Bu hücreler için devam; geri kalanlara ağırlık uygula
                dist2 = np.where(zero_mask, np.inf, dist2)

            w = 1.0 / np.power(dist2 + eps, 0.5 * p)  # 1 / d^p
            num += w * z[i]
            den += w

        # Karonun çıktısı
        Zi[j0:j1, :] = (num / np.maximum(den, eps)).astype(dtype, copy=False)
        j0 = j1

    extent = (gx0, gx1, gy0, gy1)
    # Xi, Yi'yi 1D döndürmek yeterli (imshow sadece extent kullanıyor)
    return xs, ys, Zi, extent
# =========================================================

def tri_grid(x, y, z, xmin, xmax, ymin, ymax, nx=600):
    """
    Build a regular grid (Xi, Yi, Zi, extent) via linear interpolation on a
    triangular mesh. Zi is NaN outside the convex hull (no extrapolation).
    """
    import numpy as np
    x = np.asarray(x, float); y = np.asarray(y, float); v = np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[ok], y[ok], v[ok]
    if x.size < 3:
        # not enough points to triangulate
        Xg = np.linspace(xmin, xmax, max(80, int(nx)))
        Yg = np.linspace(ymin, ymax, max(80, int(nx)))
        Xi, Yi = np.meshgrid(Xg, Yg)
        Zi = np.full_like(Xi, np.nan, dtype=float)
        return Xi, Yi, Zi, (xmin, xmax, ymin, ymax)

    Xg = np.linspace(xmin, xmax, int(nx))
    Yg = np.linspace(ymin, ymax, int(nx))
    Xi, Yi = np.meshgrid(Xg, Yg)

    tri = Triangulation(x, y)
    linI = LinearTriInterpolator(tri, v)
    Zi = np.asarray(linI(Xi, Yi), dtype="float32")  # NaN outside the hull

    return Xi, Yi, Zi, (xmin, xmax, ymin, ymax)

def _round_to_step(x, step=0.5, how="round"):
    if not np.isfinite(x): 
        return np.nan
    q = x / step
    if how == "floor":
        return step * np.floor(q)
    elif how == "ceil":
        return step * np.ceil(q)
    return step * np.round(q)

# === Projection & cartography helpers (EPSG:3310, topo base, hillshade, scalebar, N arrow) ===
CA_CRS = "EPSG:3310"   # California Albers (meters)
WGS84  = "EPSG:4326"

def to_3310_coords_df(coords_df: pd.DataFrame) -> gpd.GeoDataFrame:
    gpts = gpd.GeoDataFrame(
        coords_df.copy(),
        geometry=[Point(float(x), float(y)) for x, y in zip(coords_df["X"], coords_df["Y"])],
        crs=WGS84
    ).to_crs(CA_CRS)
    out = coords_df.copy()
    out["X_3310"] = gpts.geometry.x.values
    out["Y_3310"] = gpts.geometry.y.values
    return out

def idw_grid_3310(x_m, y_m, z, xmin, xmax, ymin, ymax, nx=320, power=2):
    xg = np.linspace(xmin, xmax, nx)
    yg = np.linspace(ymin, ymax, nx)
    Xi, Yi = np.meshgrid(xg, yg)
    Zi = np.full_like(Xi, np.nan, dtype=float)
    xs = np.asarray(x_m, float); ys = np.asarray(y_m, float); zs = np.asarray(z, float)
    mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    xs, ys, zs = xs[mask], ys[mask], zs[mask]
    if xs.size < 3: return Xi, Yi, Zi, (xmin, xmax, ymin, ymax)
    for i in range(Yi.shape[0]):
        dy2 = (yg[i] - ys) ** 2
        for j in range(Xi.shape[1]):
            dx2 = (xg[j] - xs) ** 2
            d = np.sqrt(dx2 + dy2)
            if np.any(d == 0):
                Zi[i, j] = zs[d.argmin()]
            else:
                w = 1.0 / np.power(d, power)
                Zi[i, j] = np.nansum(w * zs) / np.nansum(w)
    return Xi, Yi, Zi, (xmin, xmax, ymin, ymax)

def tri_grid_3310(x_m, y_m, z, xmin, xmax, ymin, ymax, nx=600):
    """
    Return (Xi, Yi, Zi, extent) using piecewise-linear interpolation on a Delaunay
    triangulation (matplotlib.tri). NaN outside the convex hull.

    x_m, y_m, z must be in the same CRS (here EPSG:3310 meters).
    """
    import numpy as np

    x = np.asarray(x_m, float)
    y = np.asarray(y_m, float)
    v = np.asarray(z,   float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[ok], y[ok], v[ok]
    if x.size < 3:
        # Not enough points to triangulate
        xg = np.linspace(xmin, xmax, max(80, int(nx)))
        yg = np.linspace(ymin, ymax, max(80, int(nx)))
        Xi, Yi = np.meshgrid(xg, yg)
        Zi = np.full_like(Xi, np.nan, dtype=float)
        return Xi, Yi, Zi, (xmin, xmax, ymin, ymax)

    # Regular grid for display (square pixels by default)
    xg = np.linspace(xmin, xmax, int(nx))
    yg = np.linspace(ymin, ymax, int(nx))
    Xi, Yi = np.meshgrid(xg, yg)

    tri = Triangulation(x, y)
    linI = LinearTriInterpolator(tri, v)

    Zi = linI(Xi, Yi)              # Masked array (NaN outside hull)
    Zi = np.asarray(Zi, dtype="float32")  # for imshow

    return Xi, Yi, Zi, (xmin, xmax, ymin, ymax)

def add_basemap_topo_hillshade(
    ax,
    crs="EPSG:3310",
    topo="Esri.WorldTopoMap",
    shade="Esri.WorldHillshade",
    zoom=None,
    use_hillshade=True,
    topo_alpha=0.95,
    shade_alpha=0.25,
):
    """Önce renkli topo, sonra opsiyonel hillshade; zoom=None ise zoom parametresi gönderilmez."""
    def _resolve_provider(path: str):
        prov = cx.providers
        for part in str(path).split("."):
            prov = getattr(prov, part)
        return prov

    drawn = False

    # Topo
    try:
        prov_topo = _resolve_provider(topo)
        _add_basemap_safe(ax, source=prov_topo, crs=crs, zoom=zoom, attribution=False, alpha=topo_alpha)
        drawn = True
    except Exception as e:
        print("[WARN] Topo basemap failed:", e)

    # Hillshade (opsiyonel)
    if use_hillshade:
        try:
            prov_shade = _resolve_provider(shade)
            _add_basemap_safe(ax, source=prov_shade, crs=crs, zoom=zoom, attribution=False, alpha=shade_alpha)
            drawn = True or drawn
        except Exception as e:
            print("[WARN] Hillshade failed:", e)

    # Fallbacklar
    if not drawn:
        for fb in ("CartoDB.Voyager", "OpenStreetMap.Mapnik"):
            try:
                prov_fb = _resolve_provider(fb)
                _add_basemap_safe(ax, source=prov_fb, crs=crs, zoom=zoom, attribution=False, alpha=0.95)
                drawn = True
                break
            except Exception as e:
                print(f"[WARN] Basemap fallback failed ({fb}):", e)

    return drawn

def add_basemap_color_only(
    ax,
    crs="EPSG:3310",
    topo="Esri.WorldTopoMap",
    zoom=None,
    alpha=0.95,
):
    """Sadece renkli topo (veya fallback) ekler; hillshade YOK."""
    import contextily as cx

    def _resolve_provider(path: str):
        prov = cx.providers
        for part in str(path).split("."):
            prov = getattr(prov, part)
        return prov

    def _add_basemap_safe(ax, source, crs, zoom=None, **kwargs):
        if zoom is None:
            return cx.add_basemap(ax, source=source, crs=crs, **kwargs)
        else:
            return cx.add_basemap(ax, source=source, crs=crs, zoom=int(zoom), **kwargs)

    drawn = False
    # 1) Tercih edilen renkli topo
    try:
        prov_topo = _resolve_provider(topo)
        _add_basemap_safe(ax, source=prov_topo, crs=crs, zoom=zoom, attribution=False, alpha=alpha)
        drawn = True
    except Exception as e:
        print("[WARN] Topo basemap failed:", e)

    # 2) Fallback: Carto → OSM
    if not drawn:
        for fb in ("CartoDB.Voyager", "OpenStreetMap.Mapnik"):
            try:
                prov_fb = _resolve_provider(fb)
                _add_basemap_safe(ax, source=prov_fb, crs=crs, zoom=zoom, attribution=False, alpha=alpha)
                drawn = True
                break
            except Exception as e:
                print(f"[WARN] Basemap fallback failed ({fb}):", e)

    return drawn

def add_town_labels(ax, crs=None, fontsize=8, color="#222", marker_size=3.0, z=12):
    """
    Plot major CA city labels in the axes' data CRS.
    - crs: If None, auto-try ["EPSG:3857", "EPSG:3310", "EPSG:4326"] and pick the one
           whose projected points best fall inside current view.
           If you know your map CRS, pass it explicitly (e.g., "EPSG:3857").
    NOTE: Call this AFTER basemap/limits are set so ax.get_xlim/ylim reflect the map extent.
    """
    cities_ll = [
        ("Los Angeles",   -118.2437, 34.0522),
        ("San Diego",     -117.1611, 32.7157),
        ("San Jose",      -121.8863, 37.3382),
        ("San Francisco", -122.4194, 37.7749),
        ("Fresno",        -119.7871, 36.7378),
        ("Sacramento",    -121.4944, 38.5816),
        ("Bakersfield",   -119.0187, 35.3733),
        ("Stockton",      -121.2908, 37.9577),
        ("Modesto",       -120.9969, 37.6391),
        ("Oxnard",        -119.1771, 34.1975),
        ("Riverside",     -117.3962, 33.9533),
        ("Santa Rosa",    -122.7141, 38.4405),
    ]

    # Require realistic extents
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    if not (np.isfinite([xmin, xmax, ymin, ymax]).all() and xmax > xmin and ymax > ymin):
        # nothing to do yet (probably called before basemap/limits)
        return

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception:
        # minimal fallback without GeoPandas: plot in 4326 only if the axes look like lon/lat
        is_lonlat = (abs(xmin) <= 360 and abs(xmax) <= 360 and abs(ymin) <= 90 and abs(ymax) <= 90)
        pts = []
        for n, lon, lat in cities_ll:
            if is_lonlat and (xmin <= lon <= xmax) and (ymin <= lat <= ymax):
                pts.append((n, lon, lat))
        # fractional offset
        dx, dy = xmax - xmin, ymax - ymin
        off = 0.012  # 1.2% of span
        for n, x, y in pts:
            ax.plot(x, y, marker="o", ms=marker_size, color="black", zorder=z)
            ax.text(x + off*dx, y + off*dy, n, fontsize=fontsize, color=color,
                    ha="left", va="bottom", zorder=z)
        return

    # Build LL dataframe once
    gdf_ll = gpd.GeoDataFrame(
        [(n, Point(lon, lat)) for n, lon, lat in cities_ll],
        columns=["name","geometry"], crs="EPSG:4326"
    )

    # Try candidate CRSs
    candidates = [crs] if isinstance(crs, str) else ["EPSG:3857", "EPSG:3310", "EPSG:4326"]
    best_inside = -1; best_gdf = None
    for cand in candidates:
        try:
            gdf_c = gdf_ll.to_crs(cand)
            x = gdf_c.geometry.x.values; y = gdf_c.geometry.y.values
            inside = ((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)).sum()
            if inside > best_inside:
                best_inside = inside; best_gdf = gdf_c
        except Exception:
            continue

    if best_gdf is None:
        return  # nothing worked

    # Fractional offset so it scales with the map
    dx, dy = xmax - xmin, ymax - ymin
    off = 0.012  # 1.2% of span in each direction

    for _, r in best_gdf.iterrows():
        x, y = float(r.geometry.x), float(r.geometry.y)
        if (xmin <= x <= xmax) and (ymin <= y <= ymax):
            ax.plot(x, y, marker="o", ms=marker_size, color="black", zorder=z)
            ax.text(x + off*dx, y + off*dy, r["name"], fontsize=fontsize, color=color,
                    ha="left", va="bottom", zorder=z)


# ---------------- FIGURES ----------------
# ============================================================
# STUDY / LOCATOR MAPS — rewritten, robust, label-free, color basemaps
# Requires: matplotlib.pyplot as plt, pandas as pd, geopandas as gpd,
#           shapely.geometry (Point, box), contextily as cx,
#           save_tiff(PathLike), HAS_GEO (bool), BASEMAP_PROVIDER or BASEMAP_PRIORITY (list),
#           and (opsiyonel) add_basemap_topo_hillshade(...)
# ============================================================

def _cx_resolve(path: str):
    """Resolve 'Esri.WorldTopoMap' → provider object."""
    prov = cx.providers
    for part in str(path).split("."):
        prov = getattr(prov, part)
    return prov


def plot_study_area_map(coords_df: pd.DataFrame, out_path: Path, pad_frac: float = 0.08):
    """
    Study-area map over a color basemap (optional), with an always-on quick scatter fallback.
    coords_df columns: ['Well', 'X', 'Y'] in EPSG:4326 (lon/lat).
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        import contextily as cx  # noqa: F401
    except Exception as e:
        print("[WARN] Geo stack missing; drawing simple lon/lat scatter:", e)
        fig, ax = plt.subplots(figsize=(7.6, 5.8))
        ax.scatter(coords_df["X"], coords_df["Y"], s=16, c="deepskyblue",
                   edgecolor="royalblue", linewidth=0.6, zorder=3)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(alpha=0.2)
        # adornments (approximate in lon/lat)
        add_north_arrow(ax)
        save_tiff(out_path); return

    if coords_df.empty:
        print("[WARN] plot_study_area_map: coords_df is empty."); return

    # Points -> 3857, bounds with padding
    gpts = gpd.GeoDataFrame(
        coords_df.copy(),
        geometry=[Point(float(x), float(y)) for x, y in zip(coords_df["X"], coords_df["Y"])],
        crs="EPSG:4326"
    ).to_crs(3857)

    xmin, ymin, xmax, ymax = gpts.total_bounds
    dx, dy = (xmax - xmin) * pad_frac, (ymax - ymin) * pad_frac
    xmin -= dx; xmax += dx; ymin -= dy; ymax += dy

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    # Scatter first
    gpts.plot(ax=ax, markersize=14, color="deepskyblue",
              edgecolor="royalblue", linewidth=0.7, alpha=0.95, zorder=5)

    # Optional basemap
    try:
        add_basemap_with_big_labels(ax, base="Esri.WorldTopoMap",
                                    base_zoom=12, label_zoom=13, crs="EPSG:3857")
    except Exception as e:
        print("[WARN] Basemap add failed (study area):", e)

    # Basemap (optional)
    add_basemap_with_big_labels(ax, base="Esri.WorldTopoMap",
                                base_zoom=12, label_zoom=13, crs="EPSG:3857")
    
    # >>> add town labels after limits & basemap <<<
    # add_town_labels(ax, crs="EPSG:3857", fontsize=11, marker_size=3.5, z=15)

    # Adornments in 3857
    add_scalebar(ax, length_km=50, location="lower left")
    add_north_arrow(ax, loc="upper right", fontsize=10, head_length=12)

    ax.set_axis_off()
    plt.tight_layout()
    save_tiff(out_path)

def plot_study_area_map_osm(coords_df: pd.DataFrame,
                            optional_boundary_file: Optional[str],
                            out_path: Path,
                            pad_frac: float = 0.06):
    """
    Study-area map over OSM (optional) with boundary overlay (if provided).
    Always draws point scatter immediately; basemap is added only if fast/available.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        import contextily as cx  # noqa: F401
    except Exception as e:
        print("[WARN] Geo stack missing; drawing simple lon/lat scatter:", e)
        fig, ax = plt.subplots(figsize=(7.6, 5.8))
        ax.scatter(coords_df["X"], coords_df["Y"], s=14, c="deepskyblue",
                   edgecolor="royalblue", linewidth=0.6, zorder=2)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(alpha=0.2)
        add_north_arrow(ax)
        save_tiff(out_path); return

    if coords_df.empty:
        print("[WARN] plot_study_area_map_osm: coords_df is empty."); return

    # Points -> 3857
    gpts = gpd.GeoDataFrame(
        coords_df.copy(),
        geometry=[Point(float(x), float(y)) for x, y in zip(coords_df["X"], coords_df["Y"])],
        crs="EPSG:4326"
    ).to_crs(3857)

    # Extent with padding
    xmin, ymin, xmax, ymax = gpts.total_bounds
    dx, dy = (xmax - xmin) * pad_frac, (ymax - ymin) * pad_frac
    xmin -= dx; xmax += dx; ymin -= dy; ymax += dy

    # Optional boundary
    boundary_gdf = None
    if optional_boundary_file:
        try:
            boundary_gdf = gpd.read_file(optional_boundary_file)
            if boundary_gdf.crs is None:
                boundary_gdf.set_crs(epsg=4326, inplace=True)
            boundary_gdf = boundary_gdf.to_crs(3857)
            bxmin, bymin, bxmax, bymax = boundary_gdf.total_bounds
            xmin = min(xmin, bxmin); ymin = min(ymin, bymin)
            xmax = max(xmax, bxmax); ymax = max(ymax, bymax)
        except Exception as e:
            print("[WARN] Could not read boundary file:", e)

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    # Scatter first
    gpts.plot(ax=ax, markersize=12, color="deepskyblue",
              edgecolor="royalblue", linewidth=0.7, alpha=0.95, zorder=5)

    # Boundary (if any)
    if boundary_gdf is not None and len(boundary_gdf) > 0:
        boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="black",
                          linewidth=1.0, zorder=4)

    # Basemap (optional)
    try:
        add_basemap_with_big_labels(ax, base="OpenStreetMap.Mapnik",
                                    base_zoom=12, label_zoom=13, crs="EPSG:3857")
    except Exception as e:
        print("[WARN] Basemap add failed (OSM):", e)

    add_basemap_with_big_labels(ax, base="OpenStreetMap.Mapnik",
                                base_zoom=12, label_zoom=13, crs="EPSG:3857")
    
    # >>> add town labels <<<
    # add_town_labels(ax, crs="EPSG:3857", fontsize=11, marker_size=3.5, z=15)

    # Adornments
    add_scalebar(ax, length_km=50, location="lower left")
    add_north_arrow(ax)

    ax.set_axis_off()
    plt.tight_layout()
    save_tiff(out_path)

def plot_california_locator_map(coords_df: pd.DataFrame, out_path: Path, pad_frac: float = 0.08):
    """
    California (or regional) locator map with a red study rectangle and quick scatter.
    Uses a fixed CA frame; basemap added only if fast/available.
    """
    try:
        import geopandas as gpd
        import contextily as cx  # noqa: F401
    except Exception as e:
        print("[WARN] Geo stack missing; simple lon/lat fallback:", e)
        if coords_df.empty: return
        x = coords_df["X"].astype(float); y = coords_df["Y"].astype(float)
        xmin, xmax = x.min(), x.max(); ymin, ymax = y.min(), y.max()
        dx, dy = (xmax - xmin) * pad_frac, (ymax - ymin) * pad_frac
        fig, ax = plt.subplots(figsize=(9.2, 6.8))
        ax.add_patch(plt.Rectangle((xmin - dx, ymin - dy),
                                   (xmax - xmin) + 2*dx, (ymax - ymin) + 2*dy,
                                   fill=False, edgecolor="red", linewidth=2.0, zorder=3))
        ax.scatter(x, y, s=12, color="deepskyblue", edgecolor="royalblue", linewidth=0.5, zorder=4)
        ax.set_xlim(-125.0, -114.0); ax.set_ylim(32.0, 42.5)
        add_north_arrow(ax, loc="lower left", fontsize=10, head_length=12)
        ax.set_axis_off(); plt.tight_layout(); save_tiff(out_path); return

    if coords_df.empty:
        print("[WARN] plot_california_locator_map: coords_df is empty."); return

    # Data -> 3857
    gpts = gpd.GeoDataFrame(
        coords_df.copy(),
        geometry=[Point(float(xx), float(yy)) for xx, yy in zip(coords_df["X"], coords_df["Y"])],
        crs="EPSG:4326"
    ).to_crs(3857)

    # Study rectangle (lon/lat -> 3857)
    x_ll = coords_df["X"].astype(float); y_ll = coords_df["Y"].astype(float)
    xmin_ll, xmax_ll = x_ll.min(), x_ll.max()
    ymin_ll, ymax_ll = y_ll.min(), y_ll.max()
    dx_ll, dy_ll = (xmax_ll - xmin_ll) * pad_frac, (ymax_ll - ymin_ll) * pad_frac
    rect_ll = gpd.GeoDataFrame(
        geometry=[box(xmin_ll - dx_ll, ymin_ll - dy_ll, xmax_ll + dx_ll, ymax_ll + dy_ll)],
        crs="EPSG:4326"
    ).to_crs(3857)

    # Fixed CA frame
    CA_LON_MIN, CA_LON_MAX = -125.0, -114.0
    CA_LAT_MIN, CA_LAT_MAX =  32.0,   42.5
    ca_frame = gpd.GeoSeries([box(CA_LON_MIN, CA_LAT_MIN, CA_LON_MAX, CA_LAT_MAX)], crs="EPSG:4326").to_crs(3857)
    wxmin, wymin, wxmax, wymax = ca_frame.iloc[0].bounds

    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    ax.set_xlim(wxmin, wxmax); ax.set_ylim(wymin, wymax)

    # Scatter first
    gpts.plot(ax=ax, markersize=10, color="tab:blue",
              alpha=0.95, edgecolor="white", linewidth=0.5, zorder=5)

    # Basemap (optional)
    try:
        add_basemap_with_big_labels(ax, base="Esri.WorldTopoMap",
                                    base_zoom=8, label_zoom=11, crs="EPSG:3857")
    except Exception as e:
        print("[WARN] Basemap add failed (locator):", e)

    add_basemap_with_big_labels(ax, base="Esri.WorldTopoMap",
                                base_zoom=8, label_zoom=11, crs="EPSG:3857")
    
    # >>> add town labels <<<
    # add_town_labels(ax, crs="EPSG:3857", fontsize=11, marker_size=3.5, z=15)

    # Rectangle + adornments
    rect_ll.boundary.plot(ax=ax, color="red", linewidth=2.0, zorder=6)
    add_scalebar(ax, length_km=100, location="lower left")
    add_north_arrow(ax)

    ax.set_axis_off()
    plt.tight_layout()
    save_tiff(out_path)
  
def _plot_fig2_heatmaps(OUT_FIGS: Path, df: pd.DataFrame, sgi: pd.DataFrame):
    """
    Fig.2: SGI & WSE heatmaps (same physical size as other figs).
    Uses global FIG_W, FIG_H (inches). No tight_layout(); no bbox tightening.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # --- single source of truth for size (match Fig.4) ---
    FIG_W = float(globals().get("FIG_W", 9.8))   # inches
    FIG_H = float(globals().get("FIG_H", 5.8))   # inches

    # ---------- SGI: annual mean (diverging, red=negative, blue=positive) ----------
    try:
        sgi2 = sgi.copy()
        if not isinstance(sgi2.index, pd.DatetimeIndex):
            sgi2.index = pd.to_datetime(sgi2.index, errors="coerce")
        sgi2 = sgi2.loc[sgi2.index.notna()]  # drop unparsed
        annual_sgi = sgi2.resample("YE").mean()
        if not annual_sgi.empty:
            annual_sgi.index = annual_sgi.index.year.astype(int)

            try:
                well_order = _sorted_well_order(annual_sgi.columns)
            except Exception:
                well_order = list(annual_sgi.columns)

            annual_sorted = annual_sgi.reindex(columns=well_order)
            years = annual_sorted.index.astype(int).tolist()

            # symmetric SGI range around zero
            norm = mcolors.TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=+2.5)

            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))  # << fixed size
            im = ax.imshow(
                annual_sorted.T.values,
                aspect="auto",
                interpolation="nearest",
                cmap="RdBu",
                norm=norm
            )
            ax.set_xlabel("Year"); ax.set_ylabel("Well")

            if years:
                step = max(1, len(years) // 20)
                xt = np.arange(0, len(years), step)
                ax.set_xticks(xt)
                ax.set_xticklabels([str(years[i]) for i in xt], rotation=45, ha="right")

            yt, yl = _sparse_y_labels(well_order, max_labels=24)
            ax.set_yticks(yt); ax.set_yticklabels(yl)

            cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.15)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label("SGI (red=dry, blue=wet)")
            ax.set_title("Annual mean SGI (wells × years)")

            save_tiff(OUT_FIGS / "fig2_annual_mean_heatmap.tiff")
        else:
            print("[INFO] Fig.2 SGI heatmap skipped: annual_sgi is empty or undefined.")
    except Exception as e:
        print("[INFO] Fig.2 SGI heatmap skipped:", e)

    # ---------- WSE: annual anomaly (diverging, red=negative, blue=positive) ----------
    try:
        df2 = df.copy()
        if not isinstance(df2.index, pd.DatetimeIndex):
            df2.index = pd.to_datetime(df2.index, errors="coerce")
        df2 = df2.loc[df2.index.notna()]
        annual = df2.resample("YE").mean()
        if not annual.empty:
            annual.index = annual.index.year.astype(int)

            try:
                well_order_wse = _sorted_well_order(annual.columns)
            except Exception:
                well_order_wse = list(annual.columns)

            years_avail = annual.index.tolist()
            baseline_years = years_avail[:3] if len(years_avail) >= 3 else years_avail
            baseline = annual.loc[baseline_years].mean(axis=0)
            annual_anom = annual.subtract(baseline, axis=1)

            # robust symmetric range
            flat = annual_anom.to_numpy()
            flat = flat[np.isfinite(flat)]
            if flat.size >= 10:
                q2, q98 = np.nanpercentile(flat, [2, 98])
                A = float(max(abs(q2), abs(q98)))
            else:
                A = float(np.nanmax(np.abs(flat))) if flat.size else 1.0
            if not np.isfinite(A) or A <= 0:
                A = 1.0
            norm = mcolors.TwoSlopeNorm(vmin=-A, vcenter=0.0, vmax=+A)

            annual_sorted_wse = annual_anom.reindex(columns=well_order_wse)
            years_wse = annual_sorted_wse.index.astype(int).tolist()

            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))  # << fixed size
            im = ax.imshow(
                annual_sorted_wse.T.values,
                aspect="auto",
                interpolation="nearest",
                cmap="RdBu",
                norm=norm
            )
            ax.set_xlabel("Year"); ax.set_ylabel("Well")

            if years_wse:
                step = max(1, len(years_wse) // 20)
                xt = np.arange(0, len(years_wse), step)
                ax.set_xticks(xt)
                ax.set_xticklabels([str(years_wse[i]) for i in xt], rotation=45, ha="right")

            yt, yl = _sparse_y_labels(well_order_wse, max_labels=24)
            ax.set_yticks(yt); ax.set_yticklabels(yl)

            cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.15)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label("WSE anomaly (ft; red=low, blue=high)")
            ax.set_title("Annual mean WSE anomaly (wells × years)")

            save_tiff(OUT_FIGS / "fig2_annual_mean_heatmap_WSE.tiff")
        else:
            print("[INFO] Fig.2 WSE heatmap skipped: annual is empty or undefined.")
    except Exception as e:
        print("[INFO] Fig.2 WSE heatmap skipped:", e)

def plot_sgi_trend_per_well(
    sgi: pd.DataFrame,
    out_figs: Path,
):
    """
    Fig. 3 — SGI Sen’s slope per well (colored by magnitude & sign; * marks p<0.05).
    Uses global FIG_W/FIG_H (inches) so the canvas matches other figures (e.g., Fig.4).
    Returns the built trend_df for reuse.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from tqdm import tqdm

    # --- single source of truth for size (match Fig.4) ---
    FIG_W = float(globals().get("FIG_W", 9.8))   # inches
    FIG_H = float(globals().get("FIG_H", 5.8))   # inches

    # --- Ensure datetime index
    if not isinstance(sgi.index, pd.DatetimeIndex):
        sgi = sgi.copy()
        sgi.index = pd.to_datetime(sgi.index, errors="coerce")
        sgi = sgi.loc[sgi.index.notna()]

    # --- Build trend_df
    trend_rows = []
    for w in tqdm(sgi.columns, desc="SGI trend (Sen slope) per well"):
        s = pd.to_numeric(sgi[w], errors="coerce").dropna()
        if s.empty:
            continue
        tau, p, sen_per_year = mann_kendall_sen(s)  # (tau, p, slope/year)
        trend_rows.append({
            "Well": str(w),
            "sen_slope_per_year": float(sen_per_year),
            "p_value": float(p),
            "tau": float(tau)
        })
    trend_df = pd.DataFrame(trend_rows).set_index("Well")
    if trend_df.empty:
        print("[INFO] trend_df empty; Fig.3 skipped.")
        return trend_df

    # --- Plotting (colored bars)
    try:
        well_order = _sorted_well_order(trend_df.index)
    except Exception:
        well_order = list(trend_df.index)
    trend_plot = trend_df.reindex(well_order)

    # ---- dynamic min–max domain with small padding, and matching colors ----
    vals = trend_plot["sen_slope_per_year"].to_numpy(dtype=float)
    finite = np.isfinite(vals)
    
    # 1) true min–max from data
    if finite.sum() >= 1:
        vmin = float(np.nanmin(vals[finite]))
        vmax = float(np.nanmax(vals[finite]))
    else:
        vmin, vmax = -0.5, 0.5  # safe fallback
    
    # 2) guard against degenerate span
    span = vmax - vmin
    if not np.isfinite(span) or span <= 0:
        center = vmin if np.isfinite(vmin) else 0.0
        half = max(abs(center), 0.25)
        vmin, vmax = center - half, center + half
        span = vmax - vmin
    
    # 3) proportional padding (keeps bars inside and whitespace balanced)
    pad = 0.06 * span
    x_min = vmin - pad
    x_max = vmax + pad
    
    # 4) ensure zero is inside for the diverging norm
    if x_min > 0:
        x_min = 0.0 - 0.04 * span
    if x_max < 0:
        x_max = 0.0 + 0.04 * span
    
    # 5) diverging colormap centered at 0
    cmap = mpl.colormaps.get("RdBu")   # negative=red, positive=blue
    norm = mcolors.TwoSlopeNorm(vmin=x_min, vcenter=0.0, vmax=x_max)
    
    # 6) colors for each bar using the new norm
    bar_colors = [cmap(norm(v)) if np.isfinite(v) else (0.7, 0.7, 0.7, 1.0) for v in vals]
    
    # set axis limits later with:
    # ax.set_xlim(x_min, x_max)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))  # << fixed size, matches other figs
    ax.barh(
        np.arange(len(trend_plot.index)),
        vals,
        color=bar_colors,
        edgecolor="none",
        height=0.9
    )

    # star for p<0.05
    for y, (_, row) in enumerate(trend_plot.iterrows()):
        v = row.get("sen_slope_per_year", np.nan)
        p = row.get("p_value", np.nan)
        if np.isfinite(v) and np.isfinite(p) and (p < 0.05):
            ax.plot(v, y, marker="*", markersize=7, color="black", zorder=3)

    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Sen’s slope of SGI (per year)")

    # Sparse y labels
    try:
        yticks, ylabels = _sparse_y_labels(well_order, max_labels=18)
    except Exception:
        yticks = np.linspace(0, len(well_order) - 1, min(18, len(well_order))).astype(int)
        ylabels = [well_order[i] for i in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    ax.set_title("SGI trend per well (Sen’s slope, * = p<0.05)")

    # explanatory colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.15)
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("SGI trend (per year)")

    save_tiff(out_figs / "fig3_sgi_sen_slope_per_well.tiff")
    return trend_df

def build_descriptive_stats(df_wide: pd.DataFrame,
                            coords_df: pd.DataFrame,
                            out_dir: Path = Path("./out_tables")) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tanımlayıcı istatistikleri üretir (kuyu ve havza düzeyi) ve çalışma alanı sınırlarını döndürür.
    - df_wide: Aylık WSE (geniş form) -> DatetimeIndex, sütunlar: kuyular
    - coords_df: 'Well','X','Y' (lon/lat) içeren tablo
    Dönüş: (desc_well_df, desc_basin_df, bounds_df)
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Güvenli tarih indeks ---
    if not isinstance(df_wide.index, pd.DatetimeIndex):
        try:
            df_wide = df_wide.copy()
            df_wide.index = pd.to_datetime(df_wide.index)
        except Exception:
            raise ValueError("build_descriptive_stats: df_wide index'i datetime'a çevrilemedi.")

    # --- 2) Sayısallaştırma (hatalı değerleri NaN yap) ---
    num = df_wide.apply(pd.to_numeric, errors="coerce")

    # --- 3) Kuyu bazlı istatistikler (agg/lambda yerinde elle hesap) ---
    rows = []
    for well in num.columns:
        s = num[well].dropna()
        n_nonmiss = int(s.size)
        n_total   = int(num[well].shape[0])
        n_miss    = int(n_total - n_nonmiss)

        if n_nonmiss > 0:
            s_mean  = float(s.mean())
            s_std   = float(s.std(ddof=1)) if n_nonmiss > 1 else np.nan
            s_min   = float(s.min())
            s_q1    = float(np.nanpercentile(s, 25))
            s_med   = float(np.nanmedian(s))
            s_q3    = float(np.nanpercentile(s, 75))
            s_max   = float(s.max())
            start_y = int(s.index.min().year)
            end_y   = int(s.index.max().year)
        else:
            s_mean = s_std = s_min = s_q1 = s_med = s_q3 = s_max = np.nan
            start_y = end_y = np.nan

        rows.append({
            "Well": str(well),
            "N_nonmissing": n_nonmiss,
            "N_missing": n_miss,
            "Mean": s_mean,
            "Std": s_std,
            "Min": s_min,
            "Q1": s_q1,
            "Median": s_med,
            "Q3": s_q3,
            "Max": s_max,
            "StartYear": start_y,
            "EndYear": end_y
        })

    desc_well = pd.DataFrame(rows).set_index("Well")

    # --- 4) Havza (tüm veriler) istatistikleri ---
    # Not: yeni stack uygulamasında dropna argümanı birlikte kullanılamaz; önce stack, sonra dropna.
    basin_series = num.stack(future_stack=True)
    basin_series = pd.to_numeric(basin_series, errors="coerce").dropna().astype(float)

    if basin_series.size > 0:
        b_mean = float(basin_series.mean())
        b_std  = float(basin_series.std(ddof=1)) if basin_series.size > 1 else np.nan
        b_min  = float(basin_series.min())
        b_q1   = float(np.nanpercentile(basin_series, 25))
        b_med  = float(np.nanmedian(basin_series))
        b_q3   = float(np.nanpercentile(basin_series, 75))
        b_max  = float(basin_series.max())
        n_non  = int(basin_series.size)
        n_tot  = int(num.size)  # tüm hücre sayısı (rows*cols)
        n_mis  = int(n_tot - n_non)
    else:
        b_mean = b_std = b_min = b_q1 = b_med = b_q3 = b_max = np.nan
        n_non = n_mis = n_tot = 0

    desc_basin = pd.DataFrame([{
        "N_total_cells": n_tot,
        "N_nonmissing": n_non,
        "N_missing": n_mis,
        "Mean": b_mean,
        "Std": b_std,
        "Min": b_min,
        "Q1": b_q1,
        "Median": b_med,
        "Q3": b_q3,
        "Max": b_max
    }], index=["Basin"])

    # --- 5) Çalışma alanı sınırları (lon/lat) ---
    # coords_df: Well, X(lon), Y(lat)
    try:
        xmin = float(coords_df["X"].min())
        xmax = float(coords_df["X"].max())
        ymin = float(coords_df["Y"].min())
        ymax = float(coords_df["Y"].max())
    except Exception:
        xmin = xmax = ymin = ymax = np.nan

    bounds = pd.DataFrame([{
        "min_lon": xmin, "max_lon": xmax,
        "min_lat": ymin, "max_lat": ymax,
        "width_deg": (xmax - xmin) if np.isfinite(xmin) and np.isfinite(xmax) else np.nan,
        "height_deg": (ymax - ymin) if np.isfinite(ymin) and np.isfinite(ymax) else np.nan
    }])

    # (İsteğe bağlı) diske yaz – istersen bu kısmı kaldırabilirsin
    try:
        desc_well.to_csv(out_dir / "descriptive_stats_per_well.csv")
        desc_basin.to_csv(out_dir / "descriptive_stats_basin.csv")
        bounds.to_csv(out_dir / "study_area_bounds_lonlat.csv", index=False)
    except Exception as e:
        print("[WARN] Could not write descriptive tables:", e)

    return desc_well, desc_basin, bounds

# ---------------- MAIN ----------------
def main():
    import numpy as np   # <= ensures np is bound in this scope
    
    # 0) Load CA long file
    if not Path(CALIF_FILE).exists():
        raise FileNotFoundError(f"Input not found: {CALIF_FILE}")
    df, coords_df = load_california_long(CALIF_FILE)

    # Time trimming
    if AUTO_TIME_FROM_DATA and not df.empty:
        start_eff = df.index.min(); end_eff = df.index.max()
        df = df.loc[(df.index >= start_eff) & (df.index <= end_eff)]
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.empty or df.shape[1] == 0:
        raise ValueError("No data after filtering or no well/grid columns found.")

    # --- Consistency check: wells in time series vs wells in coordinates ---
    wells_ts = pd.Index(df.columns.astype(str))
    wells_xy = pd.Index(coords_df["Well"].astype(str))
    
    missing_in_coords = wells_ts.difference(wells_xy)
    missing_in_df     = wells_xy.difference(wells_ts)
    
    if len(missing_in_coords) or len(missing_in_df):
        msg = []
        if len(missing_in_coords):
            msg.append(f"{len(missing_in_coords)} wells in df not found in coords (e.g., {list(missing_in_coords[:8])})")
        if len(missing_in_df):
            msg.append(f"{len(missing_in_df)} wells in coords not found in df (e.g., {list(missing_in_df[:8])})")
        raise ValueError("Well ID mismatch between df and coords_df: " + " | ".join(msg))
    
    # If you prefer a strict one-liner assert (less diagnostic), use:
    assert wells_ts.nunique() == len(wells_xy), \
        f"Mismatch: {wells_ts.nunique()} time-series wells vs {len(wells_xy)} coordinate wells"

    # 1) Study area & locator maps (color basemap; NO labels)
    # plot_study_area_map(coords_df, OUT_FIGS / "fig0_study_area_map.tiff")
    # plot_california_locator_map(coords_df, OUT_FIGS / "fig0b_california_locator_map.tiff")

    # 2) SGI & related tables
    sgi = calculate_sgi(df).sort_index()
    plot_sgi_trend_per_well(sgi, OUT_FIGS)

    # 2a) Per-well SGI time series plots
    # 8) SGI monthly time series for each well (with shaded severity background)
    def fill_sgi_severity_background(ax, max_abs=5.0, step=0.5, alpha_step=0.05,
                                     dry_color="tomato", wet_color="cornflowerblue"):
        """
        Shade the background in symmetric SGI bands (dry below 0, wet above 0).
        Local import ensures this works even if defined inside another function.
        """
        from matplotlib import colors as mcolors
        import numpy as np
    
        dry_rgb = mcolors.to_rgb(dry_color)
        wet_rgb = mcolors.to_rgb(wet_color)
        n_steps = int(np.ceil(max_abs / step))
    
        # dry (negative)
        for i in range(n_steps):
            y0 = -(i + 1) * step
            y1 = -i * step
            alpha = float(np.clip((i + 1) * alpha_step, 0.0, 0.6))
            ax.axhspan(y0, y1, facecolor=dry_rgb, alpha=alpha, zorder=0)
    
        # wet (positive)
        for i in range(n_steps):
            y0 = i * step
            y1 = (i + 1) * step
            alpha = float(np.clip((i + 1) * alpha_step, 0.0, 0.6))
            ax.axhspan(y0, y1, facecolor=wet_rgb, alpha=alpha, zorder=0)

    
    for well in tqdm(sgi.columns, desc="SGI series (styled)"):
        s = sgi[well]
    
        # Robust y-limits snapped to 0.5 steps
        data_min = np.nanmin([s.min(), -1.0, 0.0])
        data_max = np.nanmax([s.max(), -1.0, 0.0])
        span = data_max - data_min
        pad = 0.1 * span if (np.isfinite(span) and span > 0) else 0.5
        ymin_raw = data_min - pad
        ymax_raw = data_max + pad
        ymin = _round_to_step(ymin_raw, step=0.5, how="floor")
        ymax = _round_to_step(ymax_raw, step=0.5, how="ceil")
        if not (np.isfinite(ymin) and np.isfinite(ymax)) or (ymax <= ymin):
            ymin, ymax = -2.5, 2.5
    
        max_abs = _round_to_step(max(abs(ymin), abs(ymax)), step=0.5, how="ceil")
    
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fill_sgi_severity_background(ax, max_abs=max_abs, step=0.5, alpha_step=0.05)
    
        # SGI line and drought markers
        ax.plot(s.index, s.values, label="SGI", color="cornflowerblue", linewidth=3, zorder=1)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=1)
        ax.axhline(-1.0, color="red", linestyle="--", linewidth=1.2,
                   label="Drought threshold (SGI < -1)", zorder=1)
    
        drought_mask = s < -1.0
        ax.scatter(s.index[drought_mask], s.values[drought_mask],
                   facecolors="lightcoral", edgecolors="firebrick",
                   linewidths=0.6, s=30, label="Drought", zorder=2)
    
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("SGI (Standardized Groundwater Level)")
        ax.grid(alpha=0.3)
        
        # --- Auto-place legend where the plot is emptiest (no line coverage) ---
        # Normalize data to axes coordinates to measure point density in corners
        import matplotlib.dates as mdates
        
        # current limits (already set just above)
        ylo, yhi = ax.get_ylim()
        xnum = mdates.date2num(pd.to_datetime(s.index).to_pydatetime())
        xnorm = (xnum - xnum.min()) / max(xnum.max() - xnum.min(), 1e-12)
        ynorm = (s.values - ylo) / max(yhi - ylo, 1e-12)
        pts = np.c_[xnorm, ynorm]
        
        # candidate legend boxes in axes coords: (x0,y0)-(x1,y1)
        boxes = {
            "upper left":  ((0.02, 0.72), (0.34, 0.97)),
            "upper right": ((0.66, 0.72), (0.98, 0.97)),
            "lower left":  ((0.02, 0.03), (0.34, 0.28)),
            "lower right": ((0.66, 0.03), (0.98, 0.28)),
        }
        
        # count points inside each candidate box
        densities = {}
        for loc, ((x0, y0), (x1, y1)) in boxes.items():
            inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
            densities[loc] = int(np.sum(inside))
        
        # pick the least-occupied corner; fallback to 'best'
        best_loc = min(densities, key=densities.get) if len(densities) else "best"
        
        ax.legend(
            title=f"Well: {well}",
            loc=best_loc,
            frameon=True,
            framealpha=0.9,
            fontsize=9,
            title_fontsize=10,
        )
        
        save_tiff(OUT_SGI_SERIES / f"SGI_series_{well}.tiff")
    
    # 2b) SGI heatmap (wells × months): red = drought, with year ticks and well labels
    if isinstance(sgi.index, pd.DatetimeIndex) and (sgi.shape[0] >= 1) and (sgi.shape[1] >= 1):
        # Well order (try fixed/natural order if helper exists)
        try:
            well_order = _sorted_well_order(list(sgi.columns))
        except Exception:
            well_order = list(sgi.columns)
    
        H = sgi[well_order].copy()
        dates = H.index
    
        # Positions of year starts (if monthly data starts mid-year, use first sample of each year)
        year_first_pos = []
        year_labels    = []
        for y in np.unique(dates.year):
            pos = np.where(dates.year == y)[0]
            if pos.size:
                year_first_pos.append(int(pos[0]))
                year_labels.append(str(int(y)))
    
        # Build figure
        fig, ax = plt.subplots(figsize=(9.8, 5.8))
    
        # imshow with monthly spacing; transpose so wells are on y-axis
        im = ax.imshow(
            H.T.values,
            aspect="equal",                 # fills space; months are evenly spaced along x
            interpolation="nearest",
            cmap=SGI_CMAP,                # your red= drought palette
            vmin=-2.5, vmax=+2.5
        )
    
        # X ticks at year boundaries
        if len(year_first_pos) >= 2:
            ax.set_xticks(year_first_pos)
            ax.set_xticklabels(year_labels, rotation=45, ha="right")
            # subtle vertical lines at years
            for xp in year_first_pos:
                ax.axvline(x=xp - 0.5, color="k", alpha=0.08, linewidth=0.8)
    
        # Y ticks: sparsify well labels so they don't overlap
        try:
            yticks, ylabels = _sparse_y_labels(well_order, max_labels=24)
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels)
        except Exception:
            ax.set_yticks(np.arange(len(well_order)))
            ax.set_yticklabels(well_order)
    
        ax.set_xlabel("Year")
        ax.set_ylabel("Well")
        ax.set_title("Fig. 4 — SGI heatmap (red = drought)")
    
        # Compact colorbar on the right
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.5%", pad=0.12)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("SGI")
    
        save_tiff(OUT_FIGS / "fig4_sgi_heatmap.tiff")
    else:
        print("[WARN] Fig. 4 SGI heatmap skipped: 'sgi' is empty or index is not DatetimeIndex.")

    # =========================================
    # 2c) Drought events & yearly SGI metrics
    # =========================================
    events = []
    for col in tqdm(sgi.columns, desc="Drought events (per well)"):
        events += extract_drought_events(sgi[col], thr=-1.0)
    
    (pd.DataFrame(events)
       .to_csv(OUT_TBLS / "sgi_drought_events.csv", index=False))
    
    yearly_metrics = drought_metrics_yearly(sgi, threshold=-1.0)
    yearly_metrics.to_csv(OUT_TBLS / "drought_metrics_yearly.csv", index=False)
    
    # ==========================================================
    # 2d) Spatial maps of yearly SGI metrics — lon/lat (EPSG:4326)
    # Minimalist: NO legend, NO north arrow, NO scale bar
    # ==========================================================
    metrics = ["MaxDroughtDuration", "MinSGI", "CumulativeDeficit", "NumEvents"]
    years   = sorted(yearly_metrics["Year"].unique().tolist())
    
    # ---- fixed map extent from ALL wells (do this once) ----
    _xmin, _xmax = float(coords_df["X"].min()), float(coords_df["X"].max())
    _ymin, _ymax = float(coords_df["Y"].min()), float(coords_df["Y"].max())
    _dx, _dy     = 0.10 * (_xmax - _xmin), 0.10 * (_ymax - _ymin)  # 10% padding
    GX0, GX1, GY0, GY1 = _xmin - _dx, _xmax + _dx, _ymin - _dy, _ymax + _dy
    
    # backdrop lattice (all wells, always plotted the same)
    X_ALL = coords_df["X"].to_numpy(dtype=float)
    Y_ALL = coords_df["Y"].to_numpy(dtype=float)
    
    # ---- GLOBAL (fixed) color limits per metric, computed from ALL years ----
    limits = {}
    for metric in metrics:
        vals = yearly_metrics[metric].to_numpy(dtype=float)
    
        if metric == "MinSGI":
            # robust and safe
            try:
                lo, hi = np.nanpercentile(vals, [2, 98])
            except Exception:
                lo, hi = -2.0, 0.0
            hi = max(float(hi), 0.0)
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                lo, hi = -2.0, 0.0
        else:
            lo = 0.0
            try:
                hi = float(np.nanpercentile(vals, 98))
            except Exception:
                hi = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else 1.0
            if not (np.isfinite(hi) and hi > lo):
                hi = lo + 1.0
    
        limits[metric] = (float(lo), float(hi))
    
    # ---- draw lon/lat maps with fixed per-metric limits ----
    for metric in metrics:
        vmin, vmax = limits[metric]
    
        for y in tqdm(years, desc=f"Spatial {metric} (per year)"):
            # join with coords; wells having the metric for this year
            dfy    = yearly_metrics.loc[yearly_metrics["Year"] == y, ["Well", metric]].copy()
            merged = coords_df.merge(dfy, on="Well", how="left")
    
            # interpolation uses only valid inputs for THIS panel
            ok = (
                np.isfinite(merged[metric].to_numpy(dtype=float)) &
                np.isfinite(merged["X"].to_numpy(dtype=float)) &
                np.isfinite(merged["Y"].to_numpy(dtype=float))
            )
            if ok.sum() < 3:
                continue
    
            x_use = merged.loc[ok, "X"].to_numpy(dtype=float)
            y_use = merged.loc[ok, "Y"].to_numpy(dtype=float)
            v_use = merged.loc[ok, metric].to_numpy(dtype=float)
    
            # Interpolation (same fixed extent for every panel)
            if INTERP_METHOD.lower() == "tri":
                Xi, Yi, Zi, extent = tri_grid(
                    x_use, y_use, v_use,
                    GX0, GX1, GY0, GY1,
                    nx=TRI_NX_DEFAULT
                )
            else:
                Xi, Yi, Zi, extent = _idw_grid(
                    x_use, y_use, v_use,
                    GX0, GX1, GY0, GY1,
                    nx=IDW_NX_DEFAULT, power=IDW_POWER,
                    prefer_cell_m=IDW_PREFER_CELL_M,
                    max_pixels=IDW_MAX_PIXELS,
                    chunk_rows=IDW_CHUNK_ROWS,
                    dtype=np.float32
                )
    
            # --- Fixed (global) limits and consistent colormap choice ---
            if metric == "MinSGI":
                cmap       = "Spectral"   # lower (more negative) is worse → warmer end
                cbar_label = "Min SGI (monthly, lower = worse)"
            else:
                cmap       = "Spectral_r" # larger is worse → warmer end
                cbar_label = {
                    "MaxDroughtDuration": "Max drought duration (months)",
                    "CumulativeDeficit": "Σ(−SGI) where SGI < −1",
                    "NumEvents": "Number of drought events"
                }.get(metric, metric)
    
            # Figure size
            if "FIG_W" in globals() and "FIG_H" in globals():
                _fw, _fh = float(FIG_W), float(FIG_H)
            else:
                _fw, _fh = (24.0/2.54, 12.0/2.54)
    
            fig, ax = plt.subplots(figsize=(_fw, _fh))
    
            im = ax.imshow(
                Zi, origin="lower",
                extent=(GX0, GX1, GY0, GY1),  # fixed extent
                cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation="nearest",
                aspect="equal",
                zorder=2
            )
    
            # Backdrop: all wells (constant lattice)
            ax.scatter(X_ALL, Y_ALL, s=8, c="lightgray", edgecolor="none", zorder=3)
            # Overlay: wells with data in this panel
            ax.scatter(x_use, y_use, s=16, c="xkcd:true green",
                       edgecolor="xkcd:pine green", linewidth=0.5, zorder=4)
    
            # Clean frame
            ax.set_xlim(GX0, GX1); ax.set_ylim(GY0, GY1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.0)
    
            # Colorbar only
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)
    
            ax.set_title(f"SGI spatial metric — {metric} ({y})")
    
            save_tiff(OUT_SPATIAL / f"{metric}_{y}.tiff")
            plt.close(fig)
    
    # ==========================================================
    # 2e) SGI spatial metrics — projected (EPSG:3310), topo base
    # Minimalist: NO legend, NO north arrow, NO scale bar
    # ==========================================================
    OUT_SPATIAL_3310 = Path("./out_SGI_spatial_EN_3310")
    OUT_SPATIAL_3310.mkdir(parents=True, exist_ok=True)
    
    coords_3310 = to_3310_coords_df(coords_df)
    
    for metric in metrics:
        vmin, vmax = limits[metric]
    
        for y in tqdm(years, desc=f"Spatial {metric} (per year, 3310)"):
            dfy    = yearly_metrics[yearly_metrics["Year"] == y][["Well", metric]].copy()
            merged = coords_3310.merge(dfy, on="Well", how="left").dropna(subset=["X_3310","Y_3310",metric])
            if len(merged) < 3:
                continue
    
            xmin, xmax = merged["X_3310"].min(), merged["X_3310"].max()
            ymin, ymax = merged["Y_3310"].min(), merged["Y_3310"].max()
            dx, dy     = (xmax - xmin) * 0.10, (ymax - ymin) * 0.10
    
            _x  = merged["X_3310"].values.astype(float)
            _y  = merged["Y_3310"].values.astype(float)
            _v  = merged[metric].values.astype(float)
            gx0, gx1, gy0, gy1 = float(xmin - dx), float(xmax + dx), float(ymin - dy), float(ymax + dy)
    
            if INTERP_METHOD.lower() == "tri":
                Xi, Yi, Zi, extent = tri_grid(
                    _x, _y, _v,
                    gx0, gx1, gy0, gy1,
                    nx=TRI_NX_DEFAULT
                )
            else:
                Xi, Yi, Zi, extent = _idw_grid(
                    _x, _y, _v,
                    gx0, gx1, gy0, gy1,
                    nx=IDW_NX_DEFAULT, power=IDW_POWER,
                    prefer_cell_m=IDW_PREFER_CELL_M,
                    max_pixels=IDW_MAX_PIXELS,
                    chunk_rows=IDW_CHUNK_ROWS,
                    dtype=np.float32
                )
    
            # --- Fixed (global) limits and consistent colormap choice ---
            if metric == "MinSGI":
                cmap       = "Spectral"
                cbar_label = "Min SGI (monthly, lower = worse)"
            else:
                cmap       = "Spectral_r"
                cbar_label = {
                    "MaxDroughtDuration": "Max drought duration (months)",
                    "CumulativeDeficit": "Σ(−SGI) where SGI < −1",
                    "NumEvents": "Number of drought events"
                }.get(metric, metric)
    
            fig, ax = plt.subplots(figsize=(7.2, 5.6))
    
            # Basemap (no scale bar or north arrow)
            add_basemap_with_big_labels(
                ax,
                base="Esri.WorldTopoMap",
                base_zoom=10,
                label_zoom=48,
                crs=CA_CRS  # "EPSG:3310"
            )
    
            im = ax.imshow(Zi, origin="lower", extent=extent,
                           cmap=cmap, vmin=vmin, vmax=vmax, zorder=3)
    
            # Plot well points (warm, readable)
            ax.scatter(
                merged["X_3310"], merged["Y_3310"],
                s=16, c="xkcd:true green", edgecolor="xkcd:pine green",
                linewidth=0.6, zorder=5
            )
    
            # Frame & axes clean-up
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])
    
            # Colorbar only
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)
    
            ax.set_title(f"SGI spatial metric — {metric} ({y}) [EPSG:3310]")
    
            save_tiff(OUT_SPATIAL_3310 / f"{metric}_{y}.tiff")
            plt.close(fig)
    
    # =========================================
    # Annual matrices (always defined downstream)
    # =========================================
    try:
        annual = df.resample("YE").mean()
        annual.index = annual.index.year.astype(int)
    except Exception:
        annual = pd.DataFrame()
    
    try:
        if 'sgi' in locals() and isinstance(sgi.index, pd.DatetimeIndex):
            annual_sgi = sgi.resample("YE").mean()
            annual_sgi.index = annual_sgi.index.year.astype(int)
        else:
            annual_sgi = pd.DataFrame()
    except Exception:
        annual_sgi = pd.DataFrame()
    
    # Heatmaps (Fig. 2-like products)
    _plot_fig2_heatmaps(OUT_FIGS, df, sgi)

    # === SGI SPATIAL METRICS — projected copies with topo base (NO legend, NO north arrow, NO scale bar) ===
    OUT_SPATIAL_3310 = Path("./out_SGI_spatial_EN_3310")
    OUT_SPATIAL_3310.mkdir(parents=True, exist_ok=True)
    
    coords_3310 = to_3310_coords_df(coords_df)
    metrics = ["MaxDroughtDuration", "MinSGI", "CumulativeDeficit", "NumEvents"]
    years = sorted(yearly_metrics["Year"].unique().tolist())
    
    for metric in metrics:
        for y in tqdm(years, desc=f"Spatial {metric} (per year, 3310)"):
            dfy = yearly_metrics[yearly_metrics["Year"] == y][["Well", metric]].copy()
            merged = coords_3310.merge(dfy, on="Well", how="left").dropna(subset=["X_3310","Y_3310",metric])
            if len(merged) < 3:
                continue
    
            xmin, xmax = merged["X_3310"].min(), merged["X_3310"].max()
            ymin, ymax = merged["Y_3310"].min(), merged["Y_3310"].max()
            dx, dy = (xmax - xmin) * 0.10, (ymax - ymin) * 0.10
    
            _x = merged["X_3310"].values.astype(float)
            _y = merged["Y_3310"].values.astype(float)
            _v = merged[metric].values.astype(float)
            gx0, gx1, gy0, gy1 = float(xmin - dx), float(xmax + dx), float(ymin - dy), float(ymax + dy)
            
            if INTERP_METHOD.lower() == "tri":
                Xi, Yi, Zi, extent = tri_grid(
                    _x, _y, _v,
                    gx0, gx1, gy0, gy1,
                    nx=TRI_NX_DEFAULT
                )
            else:
                Xi, Yi, Zi, extent = _idw_grid(
                    _x, _y, _v,
                    gx0, gx1, gy0, gy1,
                    nx=IDW_NX_DEFAULT, power=IDW_POWER,
                    prefer_cell_m=IDW_PREFER_CELL_M,
                    max_pixels=IDW_MAX_PIXELS,
                    chunk_rows=IDW_CHUNK_ROWS,
                    dtype=np.float32
                )
    
            # Color scale: red = worse (consistent with earlier figures)
            data_vals = merged[metric].values.astype(float)
            if metric == "MinSGI":
                vmin = float(np.nanpercentile(data_vals, 2)); vmax = float(np.nanpercentile(data_vals, 98))
                if not np.isfinite(vmin): vmin = float(np.nanmin(data_vals)) if np.isfinite(np.nanmin(data_vals)) else -2.0
                if not np.isfinite(vmax): vmax = float(np.nanmax(data_vals)) if np.isfinite(np.nanmax(data_vals)) else 0.0
                if vmax < 0: vmax = 0.0
                if vmin >= vmax: vmin, vmax = -2.0, 0.0
                cmap = "Spectral"
                cbar_label = "Min SGI (monthly, lower = worse)"
            else:
                vmin = 0.0
                vmax = float(np.nanpercentile(data_vals, 98))
                if not np.isfinite(vmax) or vmax <= vmin:
                    vmax = float(np.nanmax(data_vals)) if np.isfinite(np.nanmax(data_vals)) else 1.0
                if vmax <= vmin: vmax = vmin + 1.0
                cmap = "Spectral_r"
                cbar_label = {
                    "MaxDroughtDuration": "Max drought duration (months)",
                    "CumulativeDeficit": "Σ(−SGI) where SGI < −1",
                    "NumEvents": "Number of drought events"
                }.get(metric, metric)
    
            fig, ax = plt.subplots(figsize=(7.2, 5.6))
    
            # Basemap (with big labels), no scale bar or north arrow
            add_basemap_with_big_labels(
                ax,
                base="Esri.WorldTopoMap",
                base_zoom=10,
                label_zoom=48,
                crs=CA_CRS  # "EPSG:3310"
            )
    
            im = ax.imshow(Zi, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, zorder=3)
    
            # Plot well points (warm, readable)
            ax.scatter(
                merged["X_3310"], merged["Y_3310"],
                s=16, c="xkcd:true green", edgecolor="xkcd:pine green", linewidth=0.6, zorder=5  
            )
    
            # Frame & axes clean-up
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])
    
            # Colorbar only (no legend)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)
    
            ax.set_title(f"SGI spatial metric — {metric} ({y}) [EPSG:3310]")
    
            # No legend, no scale bar, no north arrow
            save_tiff(OUT_SPATIAL_3310 / f"{metric}_{y}.tiff")


    # (main içinde, annual kullanılmadan hemen önce)
    df2_for_annual = df.copy()
    if not isinstance(df2_for_annual.index, pd.DatetimeIndex):
        df2_for_annual.index = pd.to_datetime(df2_for_annual.index)
    annual = df2_for_annual.resample("YE").mean()
    annual.index = annual.index.year.astype(int)

    # 3) Basin annual mean of WSE (Fig. 1) — with 95% CI (bootstrap) + LOWESS / 5-yr mean
    # Build annual means (year × well) and the basin-mean series
    annual = df.resample("YE").mean()
    annual.index = annual.index.year.astype(int)
    basin_annual_mean = annual.mean(axis=1)
    
    if len(basin_annual_mean) >= 1:
        years = basin_annual_mean.index.values
        yvals = basin_annual_mean.values
    
        # ---- Bootstrap 95% CI across wells (per year) ----
        # For each year, resample wells (with replacement) and recompute the basin mean.
        # We draw K bootstrap replicates (default 2000). Deterministic RNG for reproducibility.
        B = 2000
        rng = np.random.default_rng(42)
    
        # Precompute, for each year, the vector of valid well values
        per_year_vals = []
        for _, row in annual.iterrows():
            v = row.values.astype(float)
            v = v[np.isfinite(v)]
            per_year_vals.append(v)
    
        # If there are too few wells in a year, CI will be NaN for that year.
        boot_mat = np.full((B, len(per_year_vals)), np.nan, dtype=float)
        for b in range(B):
            col = []
            for i, v in enumerate(per_year_vals):
                k = v.size
                if k == 0:
                    col.append(np.nan)
                    continue
                # sample k wells with replacement (standard bootstrap for the mean)
                picks = rng.integers(0, k, size=k)
                col.append(np.mean(v[picks]))
            boot_mat[b, :] = np.asarray(col, dtype=float)
    
        # 95% CI from bootstrap distribution
        lo = np.nanpercentile(boot_mat, 2.5, axis=0)
        hi = np.nanpercentile(boot_mat, 97.5, axis=0)
    
        # ---- Low-frequency overlay: LOWESS (preferred) or 5-year running mean ----
        lowfreq_x = years
        lowfreq_y = None
        used_lowess = False
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            # frac controls smoothing span; ~0.35 works well for 15–25 points
            lowfreq_y = lowess(yvals, lowfreq_x, frac=0.35, return_sorted=False)
            used_lowess = True
        except Exception:
            # Fallback: centered 5-year running mean (min 3 points)
            lowfreq_y = pd.Series(yvals, index=years).rolling(5, center=True, min_periods=3).mean().values
            used_lowess = False
    
        # ---- Plot ----
        plt.figure()
        # main line (your original style)
        plt.plot(
            years, yvals,
            marker="o",
            color="steelblue",
            linewidth=1.4,
            markerfacecolor="lightskyblue",
            markeredgecolor="darkblue",
            markeredgewidth=0.8,
            label="Basin mean (annual)"
        )
    
        # 95% CI (shaded)
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            plt.fill_between(years, lo, hi, color="steelblue", alpha=0.18, linewidth=0, label="95% CI (bootstrap)")
    
        # Overlay (LOWESS or 5-yr mean)
        if lowfreq_y is not None and np.isfinite(lowfreq_y).any():
            label = "LOWESS" if used_lowess else "5-yr running mean"
            plt.plot(years, lowfreq_y, color="#1f487e", linewidth=2.0, alpha=0.9, label=label)
    
        plt.xlabel("Year")
        plt.ylabel("Groundwater level (ft a.m.s.l.)")
        title_years = f"{int(annual.index.min())}–{int(annual.index.max())}" if len(annual.index) > 0 else ""
        plt.title(f"Basin annual mean groundwater level ({REGION_LABEL}, {title_years})")
    
        # Tidy legend
        plt.legend(loc="best", frameon=True, framealpha=0.9)
        save_tiff(OUT_FIGS / "fig1_basin_annual_mean.tiff")

    # Annual mean WSE (ft) and per-well anomaly (Year − baseline)
    annual = df.resample("YE").mean()
    annual.index = annual.index.year.astype(int)
    
    # Baseline years and anomaly (reuse your global BASELINE_YEARS if defined)
    years_avail = annual.index.values.tolist()
    baseline_years = [y for y in BASELINE_YEARS if y in years_avail] if 'BASELINE_YEARS' in globals() else []
    if len(baseline_years) == 0:
        baseline_years = years_avail[:min(3, len(years_avail))]  # fallback
    
    baseline_per_well = annual.loc[baseline_years].mean(axis=0)
    annual_wse_anom  = annual.subtract(baseline_per_well, axis=1)  # same shape as 'annual'

    # === Fig. 2 — Annual mean SGI & WSE heatmaps (wide, readable) ===
    def _cm_to_in(x_cm: float) -> float:
        return float(x_cm) / 2.54
    
    def _ensure_datetime_monthly(wide: pd.DataFrame) -> pd.DataFrame:
        """Ensure DatetimeIndex; if higher freq/irregular, collapse to monthly means."""
        if not isinstance(wide.index, pd.DatetimeIndex):
            w = wide.copy()
            w.index = pd.to_datetime(w.index, errors="coerce")
            w = w.loc[w.index.notna()]
        else:
            w = wide
        # If already monthly, this is effectively a no-op; otherwise gives monthly means.
        return w.resample("MS").mean()
    
    def _order_wells(cols) -> list:
        try:
            return _sorted_well_order(cols)  # your helper if present
        except Exception:
            return list(cols)
    
    def _sparse_y(wells: list, max_labels: int = 24):
        try:
            return _sparse_y_labels(wells, max_labels=max_labels)  # your helper if present
        except Exception:
            # fallback: regular thinning
            n = len(wells)
            if n <= max_labels:
                return np.arange(n), wells
            step = max(1, n // max_labels)
            idx = np.arange(0, n, step)
            return idx, [wells[i] for i in idx]
    
    # ---------- Build annual SGI ----------
    sgi_ok = isinstance(sgi, pd.DataFrame) and not sgi.empty
    if sgi_ok:
        sgi_m = _ensure_datetime_monthly(sgi)
        # Annual mean SGI
        annual_sgi = sgi_m.resample("YE").mean()
        annual_sgi.index = annual_sgi.index.year.astype(int)
    else:
        annual_sgi = None
    
    # ---------- Build annual WSE ----------
    wse_ok = isinstance(df, pd.DataFrame) and not df.empty
    if wse_ok:
        df_m = _ensure_datetime_monthly(df)
        # Annual mean WSE
        annual_wse = df_m.resample("YE").mean()
        annual_wse.index = annual_wse.index.year.astype(int)
    else:
        annual_wse = None
    
    # ---------- Plot SGI heatmap ----------
    FIG2_W_CM, FIG2_H_CM = 24.0, 12.0  # wide for readable x-axis
    if (annual_sgi is not None) and (annual_sgi.shape[0] >= 1) and (annual_sgi.shape[1] >= 1):
        wells = _order_wells(annual_sgi.columns)
        A = annual_sgi.reindex(columns=wells)
        years = A.index.astype(int).tolist()
    
        # symmetric, robust SGI range (keep red=dry for negatives with RdBu)
        vals = A.to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size >= 10:
            # choose symmetric limit from robust 2–98% absolute range
            p2, p98 = np.nanpercentile(vals, [2, 98]).astype(float)
            vmax = float(max(abs(p2), abs(p98)))
            vmin, vmax = -vmax, +vmax
        elif vals.size >= 2:
            vmax = float(max(abs(np.nanmin(vals)), abs(np.nanmax(vals))))
            vmin, vmax = -vmax, +vmax
        else:
            vmin, vmax = -1.0, 1.0
    
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=False)
        im = ax.imshow(A.T.values, aspect="auto", interpolation="nearest",
                       cmap="RdBu", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Year"); ax.set_ylabel("Well")
    
        if len(years) > 0:
            step = max(1, len(years) // 20)  # ~≤20 x-ticks
            xt = np.arange(0, len(years), step)
            ax.set_xticks(xt)
            ax.set_xticklabels([str(years[i]) for i in xt], rotation=45, ha="right")
    
        yt, yl = _sparse_y(wells, max_labels=24)
        ax.set_yticks(yt); ax.set_yticklabels(yl)
    
        # side colorbar that doesn't squeeze axes
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax); cbar.set_label("SGI (unitless)")
        ax.set_title("Fig. 2a — Annual mean SGI (wells × years)")
        save_tiff(OUT_FIGS / "fig2_annual_mean_heatmap.tiff")
    else:
        print("[INFO] Fig.2 SGI heatmap skipped: annual_sgi is empty or undefined.")
    
    # ---------- Plot WSE heatmap ----------
    FIG2_W_CM, FIG2_H_CM = 24.0, 10.0
    if (annual_wse is not None) and (annual_wse.shape[0] >= 1) and (annual_wse.shape[1] >= 1):
        wells_wse = _order_wells(annual_wse.columns)
        W = annual_wse.reindex(columns=wells_wse)
        years = W.index.astype(int).tolist()
    
        wvals = W.to_numpy()
        wvals = wvals[np.isfinite(wvals)]
        if wvals.size >= 10:
            vmin2, vmax2 = np.nanpercentile(wvals, [2, 98]).astype(float)
            if not (np.isfinite(vmin2) and np.isfinite(vmax2) and vmax2 > vmin2):
                vmin2, vmax2 = float(np.nanmin(wvals)), float(np.nanmax(wvals))
        elif wvals.size >= 2:
            vmin2, vmax2 = float(np.nanmin(wvals)), float(np.nanmax(wvals))
        else:
            vmin2, vmax2 = -1.0, 1.0
    
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=False)
        im = ax.imshow(W.T.values, aspect="auto", interpolation="nearest",
                       cmap="YlGnBu", vmin=vmin2, vmax=vmax2)
        ax.set_xlabel("Year"); ax.set_ylabel("Well")
    
        if len(years) > 0:
            step = max(1, len(years) // 20)
            xt = np.arange(0, len(years), step)
            ax.set_xticks(xt)
            ax.set_xticklabels([str(years[i]) for i in xt], rotation=45, ha="right")
    
        yt, yl = _sparse_y(wells_wse, max_labels=24)
        ax.set_yticks(yt); ax.set_yticklabels(yl)
    
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax); cbar.set_label("WSE (ft a.m.s.l.)")
        ax.set_title("Fig. 2b — Annual mean groundwater level (wells × years)")
        save_tiff(OUT_FIGS / "fig2_annual_mean_heatmap_WSE.tiff")
    else:
        print("[INFO] Fig.2 WSE heatmap skipped: annual is empty or undefined.")
    
    # ===================== Annual WSE & anomaly maps with COMMON scales (EPSG:3310, topo base) =====================
    OUT_ANNUAL_WSE_COMMON_3310   = Path("./out_annual_gw_maps_EN_common_3310")
    OUT_ANNUAL_ANOM_COMMON_3310  = Path("./out_annual_anomaly_maps_EN_common_3310")
    OUT_ANOM_CROPPED_3310        = Path("./out_annual_anomaly_maps_EN_common_3310_cropped")  # <— NEW
    OUT_ANNUAL_WSE_COMMON_3310.mkdir(parents=True, exist_ok=True)
    OUT_ANNUAL_ANOM_COMMON_3310.mkdir(parents=True, exist_ok=True)
    OUT_ANOM_CROPPED_3310.mkdir(parents=True, exist_ok=True)  # <— NEW
    
    # Coordinates → EPSG:3310
    try:
        coords_3310 = to_3310_coords_df(coords_df)
    except Exception as e:
        print("[WARN] to_3310_coords_df failed:", e)
        coords_3310 = None
    
    # ---- ensure 'annual' exists and is numeric ----
    if isinstance(annual, pd.DataFrame) and not annual.empty:
        # enforce numeric to avoid object dtype creeping in
        annual = annual.apply(pd.to_numeric, errors="coerce")
        years = annual.index.tolist()  # list of int years
    else:
        years = []
    
    if years and (coords_3310 is not None) and (not coords_3310.empty):
        # ——— use a Well-indexed view for coordinate lookup ———
        try:
            coords_3310_idx = coords_3310.set_index("Well", drop=False)
        except Exception:
            coords_3310_idx = coords_3310.copy()
            if "Well" not in coords_3310_idx.columns:
                raise ValueError("coords_3310 does not contain a 'Well' column.")
    
        # ----- Baseline & anomaly (force numeric) -----
        try:
            years_avail = annual.index.tolist()
            baseline_years = [y for y in globals().get("BASELINE_YEARS", []) if y in years_avail]
            if len(baseline_years) == 0:
                baseline_years = years_avail[:min(3, len(years_avail))]
            baseline_per_well = pd.to_numeric(annual.loc[baseline_years].mean(axis=0), errors="coerce")
            annual_anom = annual.subtract(baseline_per_well, axis=1).apply(pd.to_numeric, errors="coerce")
        except Exception as e:
            print("[WARN] Could not compute annual anomalies:", e)
            annual_anom = pd.DataFrame()
    
        # ----- Common color ranges -----
        try:
            flat_wse = np.asarray(annual.to_numpy(), dtype=float)
            flat_wse = flat_wse[np.isfinite(flat_wse)]
            if flat_wse.size >= 10:
                WSE_VMIN, WSE_VMAX = np.nanpercentile(flat_wse, [2, 98]).astype(float)
            elif flat_wse.size >= 2:
                WSE_VMIN, WSE_VMAX = float(np.nanmin(flat_wse)), float(np.nanmax(flat_wse))
            else:
                WSE_VMIN, WSE_VMAX = 0.0, 1.0
            if not (np.isfinite(WSE_VMIN) and np.isfinite(WSE_VMAX) and (WSE_VMAX > WSE_VMIN)):
                WSE_VMIN, WSE_VMAX = 0.0, 1.0
        except Exception:
            WSE_VMIN, WSE_VMAX = 0.0, 1.0
    
        try:
            flat_an = np.asarray(annual_anom.to_numpy(), dtype=float)
            flat_an = flat_an[np.isfinite(flat_an)]
            if flat_an.size >= 10:
                a2, a98 = np.nanpercentile(flat_an, [2, 98])
                A = float(max(abs(a2), abs(a98)))
                if not (np.isfinite(A) and A > 0):
                    A = float(np.nanmax(np.abs(flat_an))) if flat_an.size else 1.0
            elif flat_an.size >= 1:
                A = float(np.nanmax(np.abs(flat_an)))
                if not (np.isfinite(A) and A > 0): 
                    A = 1.0
            else:
                A = 1.0
            ANOM_VMIN, ANOM_VMAX = -A, +A
        except Exception:
            ANOM_VMIN, ANOM_VMAX = -1.0, +1.0
    
        # ----- Grid bounds (+10% buffer) -----
        try:
            xmin = float(coords_3310["X_3310"].min()); xmax = float(coords_3310["X_3310"].max())
            ymin = float(coords_3310["Y_3310"].min()); ymax = float(coords_3310["Y_3310"].max())
            dx, dy = (xmax - xmin) * 0.10, (ymax - ymin) * 0.10
            gx0, gx1, gy0, gy1 = xmin - dx, xmax + dx, ymin - dy, ymax + dy
        except Exception as e:
            print("[WARN] 3310 bounds failed:", e)
            gx0=gx1=gy0=gy1=None
    
        if all(v is not None for v in (gx0, gx1, gy0, gy1)):
    
            # === Absolute WSE (YlGnBu), common scale ===
            for y in tqdm(years, desc="Annual WSE (common, 3310)"):
                try:
                    ok_cols = [c for c in annual.columns if c in coords_3310_idx.index]
                    if not ok_cols:
                        continue
    
                    vals = pd.to_numeric(annual.loc[y, ok_cols], errors="coerce").to_numpy(dtype=float)
                    x_m  = coords_3310_idx.loc[ok_cols, "X_3310"].to_numpy(dtype=float)
                    y_m  = coords_3310_idx.loc[ok_cols, "Y_3310"].to_numpy(dtype=float)
                    mask = np.isfinite(vals) & np.isfinite(x_m) & np.isfinite(y_m)
                    if mask.sum() < 3:
                        continue
    
                    Xi, Yi, Zi, extent = idw_grid_3310(x_m[mask], y_m[mask], vals[mask],
                                                       gx0, gx1, gy0, gy1, nx=320, power=2)
                    
                    Zi = np.asarray(Zi, dtype="float32")  # <- ensure float for imshow
    
                    fig, ax = plt.subplots(figsize=(7.2, 5.6))
                    add_basemap_color_only(ax, crs=CA_CRS)
                    im = ax.imshow(Zi, origin="lower", extent=extent, cmap="YlGnBu",
                                   vmin=WSE_VMIN, vmax=WSE_VMAX, interpolation="nearest", zorder=3)
                    ax.scatter(x_m[mask], y_m[mask], s=10, c="white",
                               edgecolor="black", linewidth=0.4, zorder=5)
                    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
                    ax.set_aspect("equal", adjustable="box")
                    add_scalebar(ax, length_km=50); add_north_arrow(ax)
    
                    cbar = plt.colorbar(im, ax=ax); cbar.set_label("WSE (ft a.m.s.l.)")
                    ax.set_title(f"Annual WSE — {y} (common scale, EPSG:3310)")
                    ax.set_xticks([]); ax.set_yticks([])
                    save_tiff(OUT_ANNUAL_WSE_COMMON_3310 / f"annual_wse_map_{y}.tiff")
                    plt.close(fig)
                except Exception as e:
                    print(f"[WARN] Annual WSE map {y} failed:", e)
    
            # === Anomaly (RdBu; red = negative), common symmetric scale ===
            # --- publication sizing ---
            FULL_W_IN, FULL_H_IN   = 7.2, 5.6     # keep as you like
            CROP_W_IN, CROP_H_IN   = 9.6, 7.2     # larger physical size for cropped-only panel
            FULL_DPI               = 400
            CROP_DPI               = 400

            for y in tqdm(years, desc="Annual WSE anomaly (common, 3310)"):
                try:
                    if annual_anom.empty:
                        break
                    ok_cols = [c for c in annual_anom.columns if c in coords_3310_idx.index]
                    if not ok_cols:
                        continue
            
                    vals = pd.to_numeric(annual_anom.loc[y, ok_cols], errors="coerce").to_numpy(dtype=float)
                    x_m  = coords_3310_idx.loc[ok_cols, "X_3310"].to_numpy(dtype=float)
                    y_m  = coords_3310_idx.loc[ok_cols, "Y_3310"].to_numpy(dtype=float)
                    mask = np.isfinite(vals) & np.isfinite(x_m) & np.isfinite(y_m)
                    if mask.sum() < 3:
                        continue
            
                    Xi, Yi, Zi, extent = idw_grid_3310(
                        x_m[mask], y_m[mask], vals[mask],
                        gx0, gx1, gy0, gy1, nx=320, power=2
                    )
                    Zi = np.asarray(Zi, dtype="float32")
                    if not np.isfinite(Zi).any():
                        continue
            
                    # ---------- FULL FIGURE (with title + colorbar) ----------
                    fig, ax = plt.subplots(figsize=(FULL_W_IN, FULL_H_IN))
                    add_basemap_color_only(ax, crs=CA_CRS)
                    im = ax.imshow(
                        Zi, origin="lower", extent=extent, cmap="RdBu",
                        vmin=ANOM_VMIN, vmax=ANOM_VMAX, interpolation="nearest", zorder=3
                    )
                    ax.scatter(x_m[mask], y_m[mask], s=10, c="white",
                               edgecolor="black", linewidth=0.4, zorder=5)
                    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
                    ax.set_aspect("equal", adjustable="box")
                    add_scalebar(ax, length_km=50); add_north_arrow(ax)
                    
                    # Year label (top-right), your style
                    import matplotlib.patheffects as pe
                    year_text = ax.text(
                        0.985, 0.985, f"{y}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=12, fontweight="bold", color="rebeccapurple", zorder=20
                    )
                    year_text.set_path_effects([pe.withStroke(linewidth=1.2, foreground="lavender")])
                    
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label("WSE anomaly (ft, baseline mean)")
                    ax.set_title(f"Annual WSE anomaly — {y} (common symmetric scale, EPSG:3310)")
                    ax.set_xticks([]); ax.set_yticks([])
                    
                    # Save full at 400 dpi
                    full_path = OUT_ANNUAL_ANOM_COMMON_3310 / f"annual_wse_anomaly_map_{y}.tiff"
                    fig.savefig(full_path, dpi=FULL_DPI, bbox_inches="tight", pad_inches=0)
                    
                    # ---------- CROPPED VARIANT (no title, no colorbar) ----------
                    try:
                        cbar.remove()
                    except Exception:
                        pass
                    ax.set_title("")  # remove title
                    
                    # Upsize the figure *before* computing the tight bbox, so the axes render larger
                    fig.set_size_inches(CROP_W_IN, CROP_H_IN)
                    
                    # Draw and compute the axes-only bounding box (in inches)
                    fig.canvas.draw()
                    renderer   = fig.canvas.get_renderer()
                    ax_bbox_in = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
                    
                    crop_path = OUT_ANOM_CROPPED_3310 / f"annual_wse_anomaly_map_{y}.tiff"
                    fig.savefig(crop_path, dpi=CROP_DPI, bbox_inches=ax_bbox_in, pad_inches=0)
                    
                    plt.close(fig)
            
                except Exception as e:
                    print(f"[WARN] Annual anomaly map {y} failed:", e)
    else:
        print("[INFO] Annual WSE/anomaly common-scale maps skipped: 'annual' empty or coords_3310 missing.")

    # === Fig 5 — Drought timelines (SGI < -1.0) ===
    try:
        # ---- figure size: reuse global FIG_W/FIG_H (inches) so it matches Fig.4 ----
        _fig_w = float(globals().get("FIG_W", 9.8))   # inches (fallback if not set globally)
        _fig_h = float(globals().get("FIG_H", 5.8))   # inches
    
        # ---- guardrails ----
        if not isinstance(sgi, pd.DataFrame) or sgi.empty:
            print("[INFO] Fig.5 skipped: SGI is empty or not a DataFrame.")
        elif "extract_drought_events" not in globals():
            print("[WARN] Fig.5 skipped: extract_drought_events(...) is undefined.")
        else:
            # Ensure datetime monthly index
            sgi5 = sgi.copy()
            if not isinstance(sgi5.index, pd.DatetimeIndex):
                sgi5.index = pd.to_datetime(sgi5.index, errors="coerce")
            sgi5 = sgi5[sgi5.index.notna()]
            if sgi5.empty:
                print("[INFO] Fig.5 skipped: SGI has no valid timestamps.")
            else:
                # If not monthly, coerce to monthly means
                if pd.infer_freq(sgi5.index) not in ("M", "MS", "ME"):
                    sgi5 = sgi5.resample("M").mean()
    
                # snap to month-end for consistent positions
                dates = sgi5.index.to_period("M").to_timestamp("M")
                if len(dates) < 2 or sgi5.shape[1] < 1:
                    print("[INFO] Fig.5 skipped: insufficient SGI data.")
                else:
                    # natural/alphanumeric order (fallback to columns)
                    try:
                        well_order5 = _sorted_well_order(sgi5.columns)
                    except Exception:
                        well_order5 = list(sgi5.columns)
    
                    # collect drought events (SGI < -1.0) per well
                    evs = []
                    for col in sgi5.columns:
                        s_ = pd.to_numeric(sgi5[col], errors="coerce")
                        if s_.notna().any():
                            evs.extend(extract_drought_events(s_, thr=-1.0))
                    events_df2 = pd.DataFrame(evs)
    
                    if events_df2.empty:
                        print("[INFO] Fig.5 skipped: no SGI<-1 events found.")
                    else:
                        # map timestamp -> x position
                        date_to_pos = {d: i for i, d in enumerate(dates)}
                        def _pos_or_nearest(ts):
                            t = pd.Timestamp(ts).to_period("M").to_timestamp("M")
                            p = date_to_pos.get(t, None)
                            if p is not None:
                                return p
                            idx = dates.get_indexer([t], method="nearest")[0]
                            return int(max(0, min(idx, len(dates)-1)))
    
                        # well -> row index
                        well_to_row = {w: i for i, w in enumerate(well_order5)}
                        # January positions for x ticks
                        year_starts = np.where(dates.month == 1)[0]
    
                        fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))  # <-- fixed size, no constrained_layout
    
                        # draw each drought event as a crimson bar
                        for _, ev in events_df2.iterrows():
                            w = ev.get("well", None)
                            if w not in well_to_row:
                                continue
                            y = well_to_row[w]
                            s_ = ev.get("start", None); e_ = ev.get("end", None)
                            if pd.isna(s_) or pd.isna(e_):
                                continue
                            s_ = pd.Timestamp(s_); e_ = pd.Timestamp(e_)
                            if e_ < s_:
                                continue
                            xs = _pos_or_nearest(s_); xe = _pos_or_nearest(e_)
                            span = int(xe - xs + 1)
                            if span <= 0:
                                continue
                            ax.broken_barh(
                                [(xs, span)], (y - 0.4, 0.8),
                                facecolor="crimson", edgecolor="crimson", alpha=0.85
                            )
    
                        # limits
                        ax.set_ylim(-1, len(well_order5))
                        ax.set_xlim(0, max(len(dates)-1, 1))
    
                        # sparse well labels
                        try:
                            yt, yl = _sparse_y_labels(well_order5, max_labels=18)
                        except Exception:
                            yt = np.linspace(0, len(well_order5)-1, min(18, len(well_order5))).astype(int)
                            yl = [well_order5[i] for i in yt]
                        ax.set_yticks(yt); ax.set_yticklabels(yl)
    
                        # year ticks (Januarys)
                        if len(year_starts) >= 1:
                            ax.set_xticks(year_starts)
                            ax.set_xticklabels([str(dates[i].year) for i in year_starts],
                                               rotation=45, ha="right")
    
                        ax.set_xlabel("Time")
                        ax.set_ylabel("Well")
                        ax.set_title("Groundwater drought timelines (SGI < -1)")
                        ax.grid(axis="x", alpha=0.15)
    
                        save_tiff(OUT_FIGS / "fig5_drought_timelines.tiff")
    except Exception as _e:
        print("[WARN] Fig.5 block failed:", _e)

    # === Fig 6 — Annual change (Δ) boxplots by year ===
    # Assumes df is monthly WSE (ft) in wide form (index: Timestamp, columns: Well)
    
    # If not already computed earlier:
    annual = df.resample("YE").mean()
    annual.index = annual.index.year.astype(int)
    annual_delta = annual.diff()  # year-over-year change per well (ft)
    
    # Keep only years where at least one well has a finite Δ
    valid_years = [int(y) for y in annual_delta.index if np.isfinite(annual_delta.loc[y].values).any()]
    data_for_box = [annual_delta.loc[y].dropna().values.astype(float) for y in valid_years]
    
    if len(valid_years) >= 1 and len(data_for_box) == len(valid_years):
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        bp = ax.boxplot(
            data_for_box,
            tick_labels=[str(y) for y in valid_years],   # use 'labels' (not 'tick_labels')
            showfliers=False,
            patch_artist=True
        )
    
        # light fill for readability
        for bx in bp['boxes']:                      # renamed from 'box' -> 'bx'
            bx.set_facecolor('#dbe6f5')
            bx.set_edgecolor('steelblue')
        for med in bp['medians']:
            med.set_color('darkblue')
            med.set_linewidth(1.4)
    
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual change Δ WSE (ft)")
        ax.set_title("Distribution of annual groundwater WSE change across wells")
        ax.grid(axis='y', alpha=0.25)
        save_tiff(OUT_FIGS / "fig6_annual_change_boxplots.tiff")

    # === Fig 7 — Basin monthly climatology (mean) with IQR across wells ===
    # df: monthly WSE (ft a.m.s.l.), index = Timestamp, columns = Well
    import calendar
    
    # Month abbreviations (use existing MON_ABBR if you already defined it)
    try:
        MON_ABBR
    except NameError:
        MON_ABBR = [calendar.month_abbr[m] for m in range(1, 13)]
    
    if df.shape[0] >= 12 and df.shape[1] >= 1:
        months = np.arange(1, 13, dtype=int)
    
        # Basin monthly mean time series (mean across wells at each timestamp)
        monthly_basin = df.mean(axis=1)
    
        # Climatological mean per calendar month (average across years)
        clim_mean = [monthly_basin[monthly_basin.index.month == m].mean() for m in months]
    
        # IQR across wells for each calendar month (25th–75th of per-well monthly means)
        iqr_low, iqr_high = [], []
        for m in months:
            # For month m, take the mean WSE per well across all years, then IQR across wells
            sub = df[df.index.month == m]                      # (months x wells)
            if sub.empty:
                iqr_low.append(np.nan); iqr_high.append(np.nan)
                continue
            per_well_mean = sub.mean(axis=0)                   # (wells,)
            if per_well_mean.size == 0 or np.all(np.isnan(per_well_mean.values)):
                iqr_low.append(np.nan); iqr_high.append(np.nan)
            else:
                q25, q75 = np.nanpercentile(per_well_mean.values, [25, 75])
                iqr_low.append(q25); iqr_high.append(q75)
    
        # Plot
        fig, ax = plt.subplots(figsize=(6.4, 3.9))
        ax.plot(months, clim_mean, marker="o",
                color="steelblue", linewidth=1.4,
                markerfacecolor="lightskyblue",
                markeredgecolor="darkblue", markeredgewidth=0.8)
    
        if not (np.all(np.isnan(iqr_low)) or np.all(np.isnan(iqr_high))):
            ax.fill_between(months, iqr_low, iqr_high, alpha=0.25, color="lightskyblue")
    
        ax.set_xticks(months)
        ax.set_xticklabels(MON_ABBR)
        ax.set_xlabel("Month")
        ax.set_ylabel("Groundwater level (ft a.m.s.l.)")
        ax.set_title("Basin monthly WSE climatology (mean) with IQR across wells")
        ax.grid(axis="y", alpha=0.25)
    
        save_tiff(OUT_FIGS / "fig7_monthly_climatology_iqr.tiff")
    
    # === Fig. 10 — Sen slope (ft/year) trend map with equal x/y scales ===
    # Input assumptions:
    #   - df           : monthly WSE (ft) wide, index=Timestamp, columns=Well
    #   - coords_df    : columns ['Well','X','Y'] (lon/lat in EPSG:4326)
    #   - _idw_grid    : helper(x, y, z, xmin, xmax, ymin, ymax, nx=..., power=...)
    #   - OUT_FIGS     : Path for figures
    #   - save_tiff    : helper to save and close a figure
    #   - tqdm         : progress bar (already imported)
    
    # 1) Annual means per well
    annual = df.resample("YE").mean()
    annual.index = annual.index.year.astype(int)
    
    # 2) Per-well trend: Sen slope (ft/year) + MK statistics
    yrs = annual.index.values.astype(float)
    trend_rows_depth = []
    for well in tqdm(annual.columns, desc="Trend (annual WSE) per well"):
        yy = annual[well].values.astype(float)
        if np.isfinite(yy).sum() >= 3:
            # Sen slope (per year)
            n = len(yy)
            slopes = []
            for i in range(n - 1):
                dy = yy[i + 1:] - yy[i]
                dt = (yrs[i + 1:] - yrs[i])
                valid = dt != 0
                if np.any(valid):
                    slopes.append(dy[valid] / dt[valid])
            sen = float(np.nanmedian(np.concatenate(slopes))) if len(slopes) else np.nan
            # Kendall tau (using order index so time spacing doesn't affect tau)
            idx = np.isfinite(yy)
            tau, p = kendalltau(np.arange(np.sum(idx)), yy[idx]) if np.sum(idx) >= 3 else (np.nan, np.nan)
        else:
            sen, tau, p = np.nan, np.nan, np.nan
        trend_rows_depth.append({"Well": str(well), "slope_ft_per_year": sen, "tau": tau, "p": p})
    
    tr_df = pd.DataFrame(trend_rows_depth)
    tr_df.to_csv(OUT_FIGS / "fig10_trend_table.csv", index=False, float_format="%.6f")
    
    # 3) Join with coordinates
    pts = coords_df.merge(tr_df, on="Well", how="left").dropna(subset=["X", "Y", "slope_ft_per_year"])
    if len(pts) >= 3:
        # Grid bounds (with padding) for IDW
        xmin, xmax = float(coords_df["X"].min()), float(coords_df["X"].max())
        ymin, ymax = float(coords_df["Y"].min()), float(coords_df["Y"].max())
        dx, dy = (xmax - xmin) * 0.10, (ymax - ymin) * 0.10
    
        Xi, Yi, Zi, extent = _idw_grid(
            pts["X"].values, pts["Y"].values, pts["slope_ft_per_year"].values,
            xmin - dx, xmax + dx, ymin - dy, ymax + dy, nx=300, power=2
        )
    
        # Robust color limits
        svals = pts["slope_ft_per_year"].values.astype(float)
        vmin = float(np.nanpercentile(svals, 2))
        vmax = float(np.nanpercentile(svals, 98))
        if not np.isfinite(vmin): vmin = float(np.nanmin(svals)) if np.isfinite(np.nanmin(svals)) else -0.2
        if not np.isfinite(vmax): vmax = float(np.nanmax(svals)) if np.isfinite(np.nanmax(svals)) else  0.2
        if vmin >= vmax: vmin, vmax = -0.2, 0.2
    
        # 4) Plot with equal x/y scales (true-to-ground placement)
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        im = ax.imshow(
            Zi, origin="lower", extent=extent,
            cmap="YlOrRd_r", vmin=vmin, vmax=vmax,
            interpolation="nearest",
            aspect="equal"  # equal scaling
        )
    
        # lock axis limits to data extent and force equal aspect
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
    
        # MK significance markers (p<0.05)
        sig = (pts["p"] < 0.05) & np.isfinite(pts["p"])
        ax.scatter(pts.loc[sig, "X"], pts.loc[sig, "Y"],
                   marker="+", s=80, c="black", linewidths=1.4, zorder=4)
        # All wells as white dots (no labels)
        ax.scatter(pts["X"], pts["Y"], s=26, c="white",
                   edgecolor="black", linewidth=0.8, zorder=3)
    
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Sen slope (ft/year)")
        ax.set_title("Fig. 10 — Sen slope (ft/year) [IDW] and MK significance (p<0.05: +)")
    
        save_tiff(OUT_FIGS / "fig10_trend_map.tiff")
    else:
        print("[WARN] Fig. 10: not enough points to map trends (need ≥3).")

    # === Fig. 11 — GW-NDSPI (sum of SGI < -1) — IDW, fixed range [-54, -32], blue = bad, red = good ===
    if isinstance(sgi.index, pd.DatetimeIndex):
        sgi_period = sgi.loc[f"{int(df.index.min().year)}-01-01": f"{int(df.index.max().year)}-12-31"]
    else:
        sgi_period = sgi.copy()
    
    # Keep NDSPI NEGATIVE: sum SGI where SGI < -1 (others → 0)
    ndspi_vals = sgi_period.where(sgi_period < -1.0, 0.0).sum(axis=0).astype(float)
    
    # Build a tidy table of NDSPI values using raw well labels
    df_ndspi = pd.DataFrame({
        "Well": ndspi_vals.index.astype(str),   # align dtypes only (no normalization)
        "Value": ndspi_vals.values
    })
    
    # Merge to coordinates using the same raw labels
    merged = coords_df.copy()
    merged["Well"] = merged["Well"].astype(str)  # dtype align only
    merged = merged.merge(df_ndspi, on="Well", how="left")
    
    # Keep rows that have coordinates AND a computed NDSPI value
    pts_b = merged.dropna(subset=["X", "Y", "Value"])
    
    if len(pts_b) >= 3:
        xmin, xmax = float(coords_df["X"].min()), float(coords_df["X"].max())
        ymin, ymax = float(coords_df["Y"].min()), float(coords_df["Y"].max())
        dx, dy = (xmax - xmin) * 0.10, (ymax - ymin) * 0.10
    
        # IDW grid
        Xi, Yi, Zi, extent = _idw_grid(
            pts_b["X"].values, pts_b["Y"].values, pts_b["Value"].values,
            xmin - dx, xmax + dx, ymin - dy, ymax + dy, nx=300, power=2
        )
    
        # ---- Fixed color range & contours
        vmin_b, vmax_b = -54.0, -32.0                 # keep within [-54, -32]
        levels_b = np.arange(vmin_b, vmax_b + 0.1, 2) # 2-unit contour spacing
    
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        im = ax.imshow(
            Zi, origin="lower", extent=extent,
            interpolation="nearest",
            cmap="RdBu", vmin=vmin_b, vmax=vmax_b
        )
    
        # Optional contours (thin, readable)
        if levels_b.size >= 2:
            try:
                cs = ax.contour(Xi, Yi, Zi, levels=levels_b, colors="k", linewidths=0.5, alpha=0.75)
                ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
            except Exception:
                pass
    
        # Points on top
        ax.scatter(pts_b["X"], pts_b["Y"], s=12, c="forestgreen",
                   edgecolor="darkgreen", linewidth=0.7, zorder=3)
    
        # Clean frame/ticks; keep black border
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.0)
    
        ax.set_title("Fig. 11 — GW-NDSPI (sum of SGI < −1; more negative = worse)")
    
        # Right colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.10)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Sum of (SGI < −1)")
    
        # Save @ 400 dpi (prefer your helper if present)
        out_path = OUT_FIGS / "fig11b_gw_ndspi_map.tiff"
        try:
            save_tiff(out_path, dpi=400)   # if your helper supports dpi kw
        except TypeError:
            plt.savefig(out_path, dpi=400, bbox_inches="tight")
            plt.close(fig)
    else:
        print("[INFO] Fig. 11 skipped: not enough points with NDSPI values.")

    # 6) Annual WSE maps (absolute vs anomalies)
    annual = df.resample("YE").mean()
    annual.index = annual.index.year.astype(int)

    # ABSOLUTE color range (robust 2–98%)
    _flat_abs = annual.values[np.isfinite(annual.values)]
    if _flat_abs.size >= 2:
        qlo, qhi = np.nanpercentile(_flat_abs, [2, 98])
        ABS_VMIN, ABS_VMAX = float(qlo), float(qhi)
    else:
        ABS_VMIN, ABS_VMAX = -1.0, 1.0
    print(f"[INFO] ABS maps range: [{ABS_VMIN:.3f}, {ABS_VMAX:.3f}]")

    # ANOMALY computation (per-well baseline)
    years_avail = annual.index.values.tolist()
    baseline_years = [y for y in BASELINE_YEARS if y in years_avail]
    if len(baseline_years) == 0:
        baseline_years = years_avail[:min(3, len(years_avail))]
    baseline_per_well = annual.loc[baseline_years].mean(axis=0)
    annual_wse_anom = annual.subtract(baseline_per_well, axis=1)

    # ANOMALY color range (symmetric, robust 2–98%)
    _flat_an = annual_wse_anom.values[np.isfinite(annual_wse_anom.values)]
    if _flat_an.size >= 2:
        qlo, qhi = np.nanpercentile(_flat_an, [2, 98])
        A = float(max(abs(qlo), abs(qhi)))
        if not (np.isfinite(A) and A > 0):
            A = float(np.nanmax(np.abs(_flat_an))) if _flat_an.size else 1.0
    else:
        A = 1.0
    ANOM_VMIN, ANOM_VMAX = -A, +A
    print(f"[INFO] ANOM maps range: [{ANOM_VMIN:.3f}, {ANOM_VMAX:.3f}]")

    # Grid bounds w/ padding for IDW
    cxmin, cxmax = float(coords_df["X"].min()), float(coords_df["X"].max())
    cymin, cymax = float(coords_df["Y"].min()), float(coords_df["Y"].max())
    pdx, pdy = 0.10 * (cxmax - cxmin), 0.10 * (cymax - cymin)
    gx0, gx1 = cxmin - pdx, cxmax + pdx
    gy0, gy1 = cymin - pdy, cymax + pdy
    
    for y in tqdm(annual.index.astype(int), desc="Annual IDW maps"):
        if USE_WSE_ANOM_FOR_ANNUAL_MAPS:
            s = annual_wse_anom.loc[y].reindex(coords_df["Well"])
            cmap = ANOM_CMAP
            # same symmetric range you already computed
            vmin, vmax = ANOM_VMIN, ANOM_VMAX
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            s = annual.loc[y].reindex(coords_df["Well"])
            # --- MAKE ABSOLUTE MAPS MATCH THE 'annual_wse_map' STYLE ---
            cmap = "YlGnBu"
            vmin, vmax = ABS_VMIN, ABS_VMAX
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
        vals = s.values.astype(float)
        mask = np.isfinite(vals)
        if mask.sum() < 3:
            continue
    
        Xi, Yi, Zi, extent = _idw_grid(
            coords_df["X"].values[mask],
            coords_df["Y"].values[mask],
            vals[mask],
            gx0, gx1, gy0, gy1,
            nx=250, power=2
        )
    
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
    
        # OPTIONAL: add the same base you used for annual_wse_map_* (if available)
        try:
            add_basemap_color_only(ax, crs=CA_CRS)
        except Exception:
            pass
    
        im = ax.imshow(
            Zi, origin="lower", extent=extent, interpolation="nearest",
            cmap=cmap, norm=norm, aspect="equal", zorder=3
        )
    
        # --- DOTS: SAME STYLE AS YOUR 'annual_wse_map_*' (white fill, black edge) ---
        ax.scatter(
            coords_df["X"].values[mask], coords_df["Y"].values[mask],
            s=14, c="white", edgecolor="black", linewidth=0.4, zorder=5
        )
    
        # same framing as the reference style
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
    
        # colorbar same label & look
        cbar = plt.colorbar(im, ax=ax)
        if USE_WSE_ANOM_FOR_ANNUAL_MAPS:
            cbar.set_label("WSE anomaly (ft, rel. to baseline)")
            title = f"Interpolated annual WSE anomaly — {y} (baseline: {baseline_years})"
            fname = OUT_ANNUAL / f"annual_wse_anom_map_{y}.tiff"
        else:
            cbar.set_label("WSE (ft a.m.s.l.)")
            title = f"Interpolated annual WSE — {y}"
            # keep your existing filename or rename for consistency:
            fname = OUT_ANNUAL / f"annual_depth_map_{y}.tiff"   # or: annual_wse_map_{y}.tiff
    
        ax.set_title(title)
        save_tiff(fname)
        plt.close(fig)

    # 7) EXTRA: a parallel set of anomaly maps for convenience (same styling)
    for y in tqdm(annual.index.astype(int), desc="Annual IDW anomaly maps (extra set)"):
        s = annual_wse_anom.loc[y].reindex(coords_df["Well"])
        vals = s.values.astype(float)
        mask = np.isfinite(vals)
        if mask.sum() < 3:
            continue
    
        # NOTE: bounds are positional (gx0, gx1, gy0, gy1)
        Xi, Yi, Zi, extent = _idw_grid(
            coords_df["X"].values[mask],
            coords_df["Y"].values[mask],
            vals[mask],
            gx0, gx1, gy0, gy1,
            nx=250,
            power=2
        )
    
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        im = ax.imshow(
            Zi, origin="lower", extent=extent, cmap=ANOM_CMAP,
            vmin=ANOM_VMIN, vmax=ANOM_VMAX, interpolation="nearest", aspect="equal"
        )
    
        # lock view to data extent + equal aspect (prevents stretch)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
    
        ax.scatter(
            coords_df["X"].values[mask], coords_df["Y"].values[mask],
            c=vals[mask], cmap=ANOM_CMAP, vmin=ANOM_VMIN, vmax=ANOM_VMAX,
            s=35, edgecolor="black"
        )
    
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("WSE anomaly (ft, rel. to baseline)")
        ax.set_title(f"Interpolated annual WSE anomaly — {y} (baseline: {baseline_years})")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(alpha=0.2)
        save_tiff(OUT_ANOMALY / f"annual_wse_anomaly_map_{y}.tiff")

    # 9) === Per-well WSE time series (separate folder, 16×8 cm @ 400 dpi) — PRO style ===
    from matplotlib import dates as mdates
    from matplotlib import ticker as mticker
    
    def export_wse_series(
        wse_wide: pd.DataFrame,
        out_dir: Path = Path("./out_WSE_series_EN"),
        fig_w_cm: float = 16.0,
        fig_h_cm: float = 8.0,
        add_roll_12: bool = True,
        add_roll_36: bool = True,
        add_baseline: bool = True,
    ) -> None:
        """
        Make per-well monthly WSE time series (publication style).
        - wse_wide: monthly WSE (wide) with DatetimeIndex, columns=Well IDs (ft a.m.s.l.)
        - out_dir: output folder (created if absent)
        """
        out_dir.mkdir(parents=True, exist_ok=True)
    
        def _cm_to_inch(x_cm: float) -> float:
            return float(x_cm) / 2.54
    
        BG_FACE   = "#FAFAFA"   # panel background
        GRID_COL  = "#E6E6E6"   # grid color
        SPINE_COL = "#444444"   # left/bottom spines
        LINE_COL  = "steelblue"
        MARKER_FC = "lightskyblue"
        MARKER_EC = "darkblue"
    
        # roll lines styling (explicit, no globals needed; you can still override via params above)
        ROLL12_COL   = "yellowgreen"
        ROLL36_COL   = "palevioletred"
        ROLL_ALPHA_12 = 0.95
        ROLL_ALPHA_36 = 0.95
        ROLL_LW_12    = 1.9
        ROLL_LW_36    = 2.1
    
        def _safe_ylim(yvals, pad_frac=0.06, fallback=(0.0, 1.0)):
            v = np.asarray(yvals, dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return fallback
            vmin, vmax = float(np.min(v)), float(np.max(v))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                return fallback
            if vmax <= vmin:
                return (vmin - 0.5, vmax + 0.5)
            pad = (vmax - vmin) * pad_frac
            return (vmin - pad, vmax + pad)
    
        def _concise_date(ax):
            # yearly majors; quarterly minors; concise formatter
            ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
            for label in ax.get_xticklabels():
                label.set(rotation=0, ha="center")
    
        # Ensure a DatetimeIndex
        if not isinstance(wse_wide.index, pd.DatetimeIndex):
            try:
                wse_wide = wse_wide.copy()
                wse_wide.index = pd.to_datetime(wse_wide.index)
            except Exception:
                print("[WARN] export_wse_series: index is not datetime-like; skipping.")
                return
    
        # Sort by time (important for rolling and plotting)
        wse_wide = wse_wide.sort_index()
    
        for well in tqdm(wse_wide.columns, desc="WSE time series (per well)"):
            try:
                s = pd.to_numeric(wse_wide[well], errors="coerce").dropna()
                if s.empty:
                    continue
    
                # Optional overlays
                roll12 = s.rolling(12, min_periods=6).mean() if add_roll_12 else None
                roll36 = s.rolling(36, min_periods=12).mean() if add_roll_36 else None
    
                # Optional baseline: mean of first 3 full calendar years
                baseline_val = None
                if add_baseline:
                    years = sorted(set(s.index.year))
                    if len(years) >= 3:
                        first3 = years[:3]
                        ss = s[(s.index.year >= first3[0]) & (s.index.year <= first3[-1])]
                        if ss.notna().sum() >= 6:
                            baseline_val = float(ss.mean())
    
                y0, y1 = int(s.index.min().year), int(s.index.max().year)
    
                fig, ax = plt.subplots(figsize=(_cm_to_inch(fig_w_cm), _cm_to_inch(fig_h_cm)))
                ax.set_facecolor(BG_FACE)
    
                # Primary monthly series
                ax.plot(
                    s.index, s.values,
                    color=LINE_COL, linewidth=1.4, marker="o", markersize=2.4,
                    markerfacecolor=MARKER_FC, markeredgecolor=MARKER_EC, markeredgewidth=0.6,
                    label="Monthly WSE"
                )
    
                # 12-month mean (mediumpurple)
                if add_roll_12 and roll12 is not None and roll12.notna().any():
                    ax.plot(
                        roll12.index, roll12.values,
                        linewidth=ROLL_LW_12, alpha=ROLL_ALPHA_12,
                        color=ROLL12_COL, label="12-mo mean", zorder=3
                    )
    
                # 36-month mean (mediumorchid)
                if add_roll_36 and roll36 is not None and roll36.notna().any():
                    ax.plot(
                        roll36.index, roll36.values,
                        linewidth=ROLL_LW_36, alpha=ROLL_ALPHA_36,
                        color=ROLL36_COL, label="36-mo mean", zorder=4
                    )
    
                # Baseline
                if baseline_val is not None and np.isfinite(baseline_val):
                    ax.axhline(baseline_val, color="#8c8c8c", linewidth=1.1, linestyle="--", label="Baseline (first 3 yrs)")
    
                # Grid & spines
                ax.grid(axis="y", color=GRID_COL, linewidth=0.8, alpha=1.0)
                ax.grid(axis="x", color=GRID_COL, linewidth=0.6, alpha=0.7)
                for side in ["top", "right"]:
                    ax.spines[side].set_visible(False)
                for side in ["left", "bottom"]:
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_color(SPINE_COL)
                    ax.spines[side].set_linewidth(1.0)
    
                # Ticks / formatters
                ax.tick_params(axis="both", which="major", length=4.0, width=0.9, colors="#222222")
                ax.tick_params(axis="both", which="minor", length=2.0, width=0.6, colors="#444444")
                _concise_date(ax)
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune=None))
    
                # Labels & limits
                ax.set_xlabel("Year")
                ax.set_ylabel("Groundwater level (ft a.m.s.l.)")
                ax.set_title(f"WSE time series — {well} ({y0}–{y1})", pad=6)
    
                ymin, ymax = _safe_ylim(s.values, pad_frac=0.06)
                ax.set_ylim(ymin, ymax)
                ax.margins(x=0.01)
    
                # Simple, robust legend (let Matplotlib choose the least-obstructive corner)
                ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=8)
    
                # Save
                save_tiff(out_dir / f"WSE_series_{well}.tiff")
                plt.close(fig)
    
            except Exception as e:
                print(f"[WARN] Failed to export WSE series for well '{well}': {e}")

    # --- Alignment & diagnostics (after df and coords_df are created) ---
    def print_well_alignment_report(df_wide: pd.DataFrame, coords: pd.DataFrame, label="GLOBAL"):
        cols = [str(c) for c in df_wide.columns]
        wcoords = [str(w) for w in coords["Well"]]
    
        n_ts = len(cols)
        n_coords_raw = len(coords)
    
        # duplicates
        dupe_mask = coords.duplicated(subset=["Well"], keep=False)
        n_dupe = int(dupe_mask.sum())
    
        # invalid coords
        import numpy as np
        inv_mask = ~np.isfinite(coords["X"].to_numpy()) | ~np.isfinite(coords["Y"].to_numpy())
        n_invalid = int(inv_mask.sum())
    
        # coverage wrt TS columns
        missing_in_coords = sorted(set(cols) - set(wcoords))
        extra_in_coords   = sorted(set(wcoords) - set(cols))
    
        print(f"[CHECK:{label}] wells in time series = {n_ts}, rows in coords = {n_coords_raw}")
        print(f"[CHECK:{label}] duplicate Well IDs in coords: {n_dupe}")
        print(f"[CHECK:{label}] invalid coord rows (non-finite X/Y): {n_invalid}")
        if missing_in_coords:
            print(f"[CHECK:{label}] TS wells missing in coords (showing up to 10): {missing_in_coords[:10]} ... ({len(missing_in_coords)} total)")
        if extra_in_coords:
            print(f"[CHECK:{label}] Extra coords not in TS (up to 10): {extra_in_coords[:10]} ... ({len(extra_in_coords)} total)")
    
    print_well_alignment_report(df, coords_df, label="POST-LOAD")

    # --- Build a plotting-safe coordinates table, aligned to df columns, no normalization ---
    coords_for_maps = (
        coords_df.copy()
        .assign(Well=lambda d: d["Well"].astype(str))
        .drop_duplicates(subset=["Well"], keep="first")
        .set_index("Well")
        .reindex([str(c) for c in df.columns])   # align to TS well order (keeps all 198 positions)
        .reset_index()
        .rename(columns={"index": "Well"})
    )
    
    # Keep only wells with finite coords for plotting (count how many are plottable)
    import numpy as np
    finite_mask = np.isfinite(coords_for_maps["X"].to_numpy()) & np.isfinite(coords_for_maps["Y"].to_numpy())
    n_plottable = int(finite_mask.sum())
    print(f"[MAP] Will plot {n_plottable} / {len(coords_for_maps)} wells with valid coordinates on study/locator maps.")

    # After df and sgi exist:
    _plot_fig2_heatmaps(OUT_FIGS, df, sgi)
    
    # ---- Safe call: only run if 'df' exists and looks right ----
    try:
        if isinstance(df, pd.DataFrame) and isinstance(df.index, (pd.DatetimeIndex, pd.Index)):
            export_wse_series(df, Path("./out_WSE_series_EN"))
        else:
            print("[INFO] Skipping export_wse_series: 'df' not ready.")
    except NameError:
        print("[INFO] 'df' is not defined yet. Call export_wse_series(your_wse_dataframe) after you build it.")

    # --- Descriptive stats (tables) ---
    desc_well, desc_basin, bounds = build_descriptive_stats(df, coords_df)
    OUT_TBLS.mkdir(parents=True, exist_ok=True)
    desc_well.to_csv(OUT_TBLS / "descriptive_stats_per_well.csv")
    desc_basin.to_csv(OUT_TBLS / "descriptive_stats_basin.csv")
    bounds.to_csv(OUT_TBLS / "study_area_bounds_lonlat.csv", index=False)

    # -- Deferred basemap maps (run last so the script "starts" instantly)
    if not ('DEFER_BASEMAPS' in globals() and DEFER_BASEMAPS):
        pass  # user opted not to defer; nothing to do
    else:
        try:
            plot_study_area_map(coords_df, OUT_FIGS / "fig0_study_area_map.tiff")
            plot_california_locator_map(coords_df, OUT_FIGS / "fig0b_california_locator_map.tiff")
        except Exception as e:
            print("[WARN] Deferred basemap maps skipped:", e)

    plot_study_area_map(coords_for_maps, OUT_FIGS / "fig0_study_area_map.tiff")
    plot_study_area_map_osm(coords_for_maps, optional_boundary_file=None,
                            out_path=OUT_FIGS / "fig0_study_area_map_OSM.tiff")
    plot_california_locator_map(coords_for_maps, OUT_FIGS / "fig0b_california_locator_map.tiff")

if __name__ == "__main__":
    main()
