"""Static configuration constants for the data pipeline.

Edit this file (not magic numbers in modules) when tuning the pipeline.
"""

from datetime import UTC, datetime

# ── Spatial extent ────────────────────────────────────────────────────────────

# Belgium bounding box. Lat is south→north; lon is west→east (positive east).
BBOX_LAT = (49.5, 51.5)
BBOX_LON = (2.5, 6.5)


# ── GFS variables ─────────────────────────────────────────────────────────────

# GFS forecast cycles are read from dynamical.org's ARCO Zarr (Apache 2.0,
# public). The store is keyed by (init_time, lead_time, latitude, longitude),
# is hourly out to 120h lead, and covers init_times back to 2021-05-01.
# Reference: https://dynamical.org/catalog/noaa-gfs-forecast/
GFS_ZARR_URL = "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr"

GFS_CYCLE_HOURS = (0, 6, 12, 18)
# With 4 cycles/day the fxx (forecast hour) values used per task are:
#   solar/wind/demand-1d-ahead (lead=24h): fxx 24..29 (latest cycle 24h before t)
GFS_FXX_RANGE = range(24, 30)

# Dynamical's GFS-forecast zarr exposes these data_vars (units in attrs):
#   temperature_2m (°C), wind_u_10m, wind_v_10m, total_cloud_cover_atmosphere (%),
#   downward_short_wave_radiation_flux_surface (W/m²), and ~15 more.
# wind10m_fcst is derived downstream as sqrt(u² + v²); cloud_cover is scaled to 0–1.
GFS_VAR_RENAME: dict[str, str] = {
    "downward_short_wave_radiation_flux_surface": "ghi_fcst",
    "temperature_2m": "t2m_fcst",
    "total_cloud_cover_atmosphere": "cloud_cover_fcst",
    # u/v handled separately — combined to wind10m_fcst magnitude
}
GFS_WIND_U_VAR = "wind_u_10m"
GFS_WIND_V_VAR = "wind_v_10m"


# ── Task metadata ─────────────────────────────────────────────────────────────

TASK_IDS = (
    "solar-1d-ahead",
    "wind-1d-ahead",
    "demand-1d-ahead-test",
)

TASK_LEAD_HOURS = {
    "solar-1d-ahead": 24,
    "wind-1d-ahead": 24,
    "demand-1d-ahead-test": 24,
}

TASK_TARGET_COLUMN = {
    "solar-1d-ahead": "solar_mwh",
    "wind-1d-ahead": "wind_mwh",
    "demand-1d-ahead-test": "demand_mwh",
}

TASK_DISPLAY_NAME = {
    "solar-1d-ahead": "Day-ahead solar energy (Belgium aggregate)",
    "wind-1d-ahead": "Day-ahead wind energy (Belgium aggregate)",
    "demand-1d-ahead-test": "Day-ahead electricity demand (Belgium total)",
}

TASK_METRIC_LABEL = {
    "solar-1d-ahead": "MAE (MWh)",
    "wind-1d-ahead": "MAE (MWh)",
    "demand-1d-ahead-test": "MAE (MWh)",
}


# ── ENTSO-E codes ─────────────────────────────────────────────────────────────

ENTSOE_AREA_BE = "10YBE----------2"  # Belgium control area
ENTSOE_PSR_SOLAR = "B16"
ENTSOE_PSR_WIND_ONSHORE = "B19"
ENTSOE_PSR_WIND_OFFSHORE = "B18"


# ── NOAA ISD ──────────────────────────────────────────────────────────────────

# Brussels Airport (EBBR). NOAA ISD recently renamed per-station CSVs from
# `<USAF>-<WBAN>.csv` (hyphenated) to `<USAF><WBAN>.csv` (concatenated).
# Using the new format; if the rename ever flips back, drop the underscore.
ISD_STATION_EBBR = "06451099999"


# ── Pipeline windows ──────────────────────────────────────────────────────────

TRAIN_START = datetime(2023, 1, 1, tzinfo=UTC)
TRAIN_END = datetime(2025, 1, 1, tzinfo=UTC)
TEST_START = datetime(2025, 1, 1, tzinfo=UTC)
TEST_END = datetime(2026, 1, 1, tzinfo=UTC)


# ── Sanity thresholds ─────────────────────────────────────────────────────────

# Maximum acceptable fraction of rows dropped during build, per task.
DROP_THRESHOLD = {
    "solar-1d-ahead": 0.01,
    "wind-1d-ahead": 0.01,
    "demand-1d-ahead-test": 0.01,
}

# Calibration sanity envelopes (min_mae, max_mae). Populated AFTER the solar
# pilot in Task 12 — until then, calibrate() warns but does not block.
CALIBRATION_ENVELOPE: dict[str, tuple[float, float]] = {}
