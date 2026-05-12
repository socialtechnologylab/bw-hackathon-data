"""Static configuration constants for the data pipeline.

Edit this file (not magic numbers in modules) when tuning the pipeline.
"""

from datetime import UTC, datetime

# ── Spatial extent ────────────────────────────────────────────────────────────

# Belgium bounding box. Lat is south→north; lon is west→east (positive east).
BBOX_LAT = (49.5, 51.5)
BBOX_LON = (2.5, 6.5)


# ── GFS variables ─────────────────────────────────────────────────────────────

# Mapping from Herbie search strings (regex over the GRIB IDX) to the eventual
# output column name in the cached parquet. UGRD/VGRD are combined to magnitude
# downstream; cloud cover is divided by 100 to land in [0, 1].
GFS_HERBIE_SEARCH = "|".join(
    [
        r":DSWRF:surface:",
        r":TMP:2 m above ground:",
        r":UGRD:10 m above ground:",
        r":VGRD:10 m above ground:",
        r":TCDC:entire atmosphere",
    ]
)

GFS_CYCLE_HOURS = (0, 6, 12, 18)
GFS_FXX_RANGE = range(0, 31)  # forecast hours 0..30 inclusive (>= max task L)


# ── Task metadata ─────────────────────────────────────────────────────────────

TASK_IDS = (
    "solar-1d-ahead",
    "wind-2h-ahead",
    "temp-1d-ahead",
    "demand-1d-ahead",
)

TASK_LEAD_HOURS = {
    "solar-1d-ahead": 24,
    "wind-2h-ahead": 2,
    "temp-1d-ahead": 24,
    "demand-1d-ahead": 24,
}

TASK_TARGET_COLUMN = {
    "solar-1d-ahead": "solar_mwh",
    "wind-2h-ahead": "wind_mwh",
    "temp-1d-ahead": "temp_c",
    "demand-1d-ahead": "demand_mwh",
}

TASK_DISPLAY_NAME = {
    "solar-1d-ahead": "Day-ahead solar energy (Belgium aggregate)",
    "wind-2h-ahead": "2-hour-ahead wind energy (Belgium aggregate)",
    "temp-1d-ahead": "Day-ahead temperature (Brussels EBBR)",
    "demand-1d-ahead": "Day-ahead electricity demand (Belgium total)",
}

TASK_METRIC_LABEL = {
    "solar-1d-ahead": "MAE (MWh)",
    "wind-2h-ahead": "MAE (MWh)",
    "temp-1d-ahead": "MAE (°C)",
    "demand-1d-ahead": "MAE (MWh)",
}


# ── ENTSO-E codes ─────────────────────────────────────────────────────────────

ENTSOE_AREA_BE = "10YBE----------2"  # Belgium control area
ENTSOE_PSR_SOLAR = "B16"
ENTSOE_PSR_WIND_ONSHORE = "B17"
ENTSOE_PSR_WIND_OFFSHORE = "B18"


# ── NOAA ISD ──────────────────────────────────────────────────────────────────

# Brussels Airport (EBBR). Format: <USAF>-<WBAN>, used in NOAA ISD filenames.
ISD_STATION_EBBR = "064510-99999"


# ── Pipeline windows ──────────────────────────────────────────────────────────

TRAIN_START = datetime(2023, 1, 1, tzinfo=UTC)
TRAIN_END = datetime(2025, 1, 1, tzinfo=UTC)
TEST_START = datetime(2025, 1, 1, tzinfo=UTC)
TEST_END = datetime(2026, 1, 1, tzinfo=UTC)


# ── Sanity thresholds ─────────────────────────────────────────────────────────

# Maximum acceptable fraction of rows dropped during build, per task.
# Temp gets a looser threshold because METAR coverage is patchier than ENTSO-E.
DROP_THRESHOLD = {
    "solar-1d-ahead": 0.01,
    "wind-2h-ahead": 0.01,
    "temp-1d-ahead": 0.03,
    "demand-1d-ahead": 0.01,
}

# Calibration sanity envelopes (min_mae, max_mae). Populated AFTER the solar
# pilot in Task 12 — until then, calibrate() warns but does not block.
CALIBRATION_ENVELOPE: dict[str, tuple[float, float]] = {}
