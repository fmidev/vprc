"""Parser for IRIS VVP (Velocity Volume Processing) prodx files."""

import logging
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from ..config import (
    _build_radar_name_to_code_map,
    _get_radar_metadata,
    _load_radar_defaults,
    resolve_radar_code,
)
from ..constants import MDS, STEP
from ..exceptions import NoUsableProfileError

logger = logging.getLogger(__name__)

# Values for profile levels absent from the VVP file. allprof_prodx2.pl stores
# levels in a height-indexed array, so a skipped level is undefined: its ZCnt
# fails every sample threshold and it is written out as MDS with zero samples.
_MISSING_LEVEL_FILL: dict[str, float] = {
    "count": 0,
    "zcount": 0,
    "lin_dbz": MDS,
    "log_dbz": MDS,
}


def _fill_missing_levels(
    ds: xr.Dataset, antenna_height_m: int, lowest_level_m: int
) -> xr.Dataset:
    """Reindex a profile onto a complete ``STEP`` grid from the lowest level.

    The grid starts at the configured lowest profile level (legacy
    ``$alintaso``), not at the lowest level present in the file, so the
    height-relative rules near the surface use the right levels. Processing
    stages index neighbouring levels as ``height ± STEP``, as the legacy
    height-indexed arrays did. Missing levels become no-echo levels with zero
    samples; other variables are NaN. Their heights above antenna are listed
    in the ``missing_levels_m`` attribute, because the legacy post-bright-band
    clutter correction treats absent levels differently from measured no-echo
    levels.

    Raises:
        ValueError: If the heights are not on the ``STEP`` grid through the
            lowest level, i.e. the radar geometry does not match the file.
        NoUsableProfileError: If no level is at or above the lowest level.
    """
    heights = ds["height"].values
    if np.any((heights - lowest_level_m) % STEP):
        raise ValueError(
            f"VVP profile levels at {(heights + antenna_height_m).tolist()} m ASL "
            f"are not on the {STEP} m grid through the lowest profile level "
            f"{lowest_level_m} m above the antenna at {antenna_height_m} m ASL; "
            "check antenna_height_m and lowest_level_offset_m"
        )
    # allprof_prodx2.pl stores levels below $alintaso at index 0, unused.
    ds = ds.isel(height=heights >= lowest_level_m)
    if not ds.sizes["height"]:
        raise NoUsableProfileError(
            "VVP profile has no levels at or above the lowest profile level "
            f"({lowest_level_m} m above the antenna)"
        )
    grid = np.arange(lowest_level_m, int(ds["height"].values[-1]) + 1, STEP)
    missing = np.setdiff1d(grid, ds["height"].values)
    if len(missing):
        logger.info(
            "VVP profile is missing %d level(s) at %s m ASL; treating them as "
            "no-echo levels with zero samples",
            len(missing),
            (missing + antenna_height_m).tolist(),
        )
        fill = {
            name: _MISSING_LEVEL_FILL.get(str(name), np.nan) for name in ds.data_vars
        }
        ds = ds.reindex(height=grid, fill_value=fill)
    ds.attrs["missing_levels_m"] = [int(height) for height in missing]
    return ds


def _whole_meters(metadata: dict, key: str) -> int:
    """Return a geometry value as integer meters, rejecting other values."""
    value = metadata[key]
    numeric = isinstance(value, Real) and not isinstance(value, bool)
    meters = float(value) if numeric else np.nan
    if not np.isfinite(meters) or not meters.is_integer():
        raise ValueError(f"{key} must be whole meters, got {value!r}")
    return int(meters)


@dataclass
class VVPHeader:
    """Metadata from VVP file header line."""

    radar: str
    product: str
    scan_type: str
    timestamp: datetime
    elevation_angles: List[float]


def _parse_vvp_header(header_line: str) -> VVPHeader:
    """
    Parse the first line of a VVP file.

    Format:
    KANKAANPAA VVP_40 PPI1_A 2025 08 24 11 00 ELEVS:  0.7 1.5 3.0

    Args:
        header_line: First line from VVP file

    Returns:
        VVPHeader object with parsed metadata

    Raises:
        ValueError: If header format is invalid
    """
    parts = header_line.strip().split()

    if len(parts) < 9:
        raise ValueError(
            f"Invalid header: expected at least 9 fields, got {len(parts)}"
        )

    radar = parts[0]
    product = parts[1]
    scan_type = parts[2]
    year = int(parts[3])
    month = int(parts[4])
    day = int(parts[5])
    hour = int(parts[6])
    minute = int(parts[7])

    # Find ELEVS: keyword and extract elevation angles
    if "ELEVS:" in parts:
        elevs_idx = parts.index("ELEVS:")
        elevation_angles = [float(x) for x in parts[elevs_idx + 1 :]]
        if not all(np.isfinite(elevation_angles)):
            raise ValueError("Elevation angles must be finite")
    else:
        elevation_angles = []

    timestamp = datetime(year, month, day, hour, minute)

    return VVPHeader(
        radar=radar,
        product=product,
        scan_type=scan_type,
        timestamp=timestamp,
        elevation_angles=elevation_angles,
    )


def _parse_vvp_file(filepath: Path | str) -> Tuple[VVPHeader, pd.DataFrame]:
    """
    Parse a VVP vertical profile file.

    File format:
        Line 1: Header with radar, product, timestamp, elevations
        Line 2: Blank
        Line 3: Column names
        Lines 4+: Data (space-separated, right-aligned numeric columns)

    Args:
        filepath: Path to VVP prodx file

    Returns:
        Tuple of (header metadata, DataFrame with profile data)

    Raises:
        NoUsableProfileError: If a valid header is followed by no data rows.
        ValueError: If the file is empty or malformed.

    Example:
        >>> header, df = parse_vvp_file('202508241100_KAN.VVP_40.txt')
        >>> header.radar
        'KANKAANPAA'
        >>> df.columns
        Index(['Height', 'Count', 'ZCnt', 'Wind-Speed', ...])
        >>> df['Height'].min(), df['Height'].max()
        (100, 6900)
    """
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty VVP file: {filepath}")

    # Parse header (line 0)
    header = _parse_vvp_header(lines[0])
    if not any(line.strip() for line in lines[3:]):
        # IRIS writes header-only products when no profile could be computed.
        raise NoUsableProfileError(f"VVP file has no profile levels: {filepath}")

    # The file has a multi-level header structure where each major column
    # (Wind-Speed, Linear-dBZ, etc.) has 2 sub-values (mean and std deviation)
    # We'll use read_fwf with colspecs or just treat it as whitespace-delimited
    # and manually assign proper column names

    # Strategy: Skip header lines, read all numeric columns, then assign names
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=3,  # Skip header, blank, and column name line
        header=None,  # No header in data
        engine="python",
    )

    # Assign column names based on the structure:
    # Height Count ZCnt  Wind-Speed(2) Direction(2) Vertical(2) Linear-dBZ(2) Log-dBZ(2)
    # Divergence(2) Deformation(2) Axis-of-Dil(2)
    # Total: 3 + 2*8 = 19 columns

    expected_cols = [
        "height",
        "count",
        "zcount",
        "wind_speed",
        "wind_speed_std",
        "direction",
        "direction_std",
        "vertical",
        "vertical_std",
        "lin_dbz",
        "lin_dbz_std",
        "log_dbz",
        "log_dbz_std",
        "divergence",
        "divergence_std",
        "deformation",
        "deformation_std",
        "axis_of_dil",
        "axis_of_dil_std",
    ]

    if len(df.columns) != len(expected_cols):
        raise ValueError(
            f"Expected {len(expected_cols)} columns in {filepath}, got {len(df.columns)}"
        )
    df.columns = expected_cols
    if df.empty:
        raise ValueError(f"No profile data in {filepath}")
    for column in ("height", "count", "zcount", "lin_dbz"):
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column]).all():
            raise ValueError(f"Non-finite or missing {column} values in {filepath}")

    # Ensure height is integer
    if (df["height"] % 1 != 0).any():
        raise ValueError(f"Profile heights must be integer meters in {filepath}")
    df["height"] = df["height"].astype(int)
    if df["height"].duplicated().any():
        raise ValueError(f"Duplicate profile heights in {filepath}")

    # Sort by height ascending (files go high->low)
    df = df.sort_values("height").reset_index(drop=True)

    return header, df


def _vvp_dataframe_to_xarray(
    df: pd.DataFrame,
    header: VVPHeader,
    radar_metadata: dict | None = None,
    source_file: Path | str | None = None,
) -> xr.Dataset:
    """
    Convert parsed VVP DataFrame to xarray Dataset.

    Heights are converted from sea level (as in VVP files) to heights above
    antenna, which is the appropriate coordinate system for VPR correction
    since we're dealing with observation geometry.

    Args:
        df: DataFrame from parse_vvp_file (heights in meters ASL)
        header: VVPHeader metadata
        radar_metadata: Optional dict to override specific metadata fields.
                       Common fields:
                       - antenna_height_m: Antenna elevation above sea level (meters)
                       - lowest_level_offset_m: Lowest profile level above the
                         antenna (meters); override together with antenna_height_m
                       - freezing_level_m: Freezing level in meters ASL (from NWP)
                         * None: Unknown (not yet retrieved from NWP)
                         * 0 or negative: No freezing layer → normalized to 0
                         * Positive: Valid freezing level (converted to above-antenna internally)

                       If not provided or partially provided, missing fields will be
                       loaded from TOML configuration (env var or package default).
        source_file: Path to source VVP file for metadata

    Returns:
        xarray.Dataset with dimensions (height,) where height is meters
        above antenna. Rich metadata includes antenna_height_m for converting
        back to sea level if needed.

    Note:
        For multi-profile processing (multiple scans), you'd extend this
        to have dimensions (height, profile_idx).
    """
    # Resolve metadata with proper precedence. An unconfigured header name is
    # an error: [defaults] would silently shift every level of the profile.
    radar_code = resolve_radar_code(header.radar)

    # Get merged metadata (TOML defaults + overrides)
    metadata = _get_radar_metadata(radar_code, radar_metadata)

    # Convert sea-level heights to heights above antenna
    # This follows the legacy Perl logic: $Height = $SeaHeight - $antennin_korkeus * 1000
    # (allprof_prodx2.pl line 153)
    antenna_height_m = _whole_meters(metadata, "antenna_height_m")
    lowest_level_m = _whole_meters(metadata, "lowest_level_offset_m")
    if lowest_level_m <= 0:
        raise ValueError(
            f"lowest_level_offset_m must be above the antenna, got {lowest_level_m}"
        )
    df = df.copy()
    df["height"] = df["height"] - antenna_height_m

    # Drop levels at or below the antenna (non-positive heights)
    df = df[df["height"] > 0].copy()
    if df.empty:
        raise NoUsableProfileError("VVP profile has no levels above the antenna")

    # Convert DataFrame to xarray using height as the index/coordinate
    ds = _fill_missing_levels(
        df.set_index("height").to_xarray(), antenna_height_m, lowest_level_m
    )

    # Rename 'count' to 'sample_count' for consistency with processing modules
    if "count" in ds:
        ds = ds.rename({"count": "sample_count"})
        ds["sample_count"].attrs["long_name"] = "Wind sample count (Count)"
    if "zcount" in ds:
        ds["zcount"].attrs["long_name"] = "Reflectivity sample count (ZCnt)"

    # Add corrected_dbz as a copy of lin_dbz (will be modified by processing)
    ds["corrected_dbz"] = ds["lin_dbz"].copy()

    # Build metadata attributes
    attrs = {
        "radar": header.radar,
        "radar_code": radar_code,
        "product": header.product,
        "scan_type": header.scan_type,
        "timestamp": header.timestamp.isoformat(),
        "elevation_angles": header.elevation_angles,
        "source_file": str(source_file) if source_file else "unknown",
    }

    # Add radar-specific metadata
    # Handle freezing_level_m specially with three cases:
    # 1. None or not provided: Unknown (not yet retrieved from NWP) → keep as None
    # 2. Zero or negative (ASL or above-antenna): No melting layer visible → set to 0
    # 3. Positive above-antenna: Valid freezing level for BB detection
    # (following Perl logic from allprof_prodx2.pl lines 359-362)
    metadata_copy = metadata.copy()
    if "freezing_level_m" in metadata_copy:
        fl = metadata_copy["freezing_level_m"]
        if fl is not None:
            if not isinstance(fl, Real) or isinstance(fl, bool) or not np.isfinite(fl):
                raise ValueError("freezing_level_m must be finite meters ASL or None")
            if fl <= 0:
                # Explicitly no freezing layer (or below surface)
                metadata_copy["freezing_level_m"] = 0
            else:
                # Convert from ASL to above-antenna (same as height coordinate)
                # If result is ≤0, freezing level is at/below antenna → no melting visible
                metadata_copy["freezing_level_m"] = (
                    fl - antenna_height_m if float(fl) > antenna_height_m else 0
                )

    attrs.update(metadata_copy)

    ds.attrs.update(attrs)

    return ds


def read_vvp(filepath: Path | str, radar_metadata: dict | None = None) -> xr.Dataset:
    """
    Parse VVP file directly to xarray Dataset (high-level wrapper).

    This is the recommended API for simple use cases.

    Args:
        filepath: Path to VVP prodx file
        radar_metadata: Optional dict to override specific metadata fields.
                       Configuration precedence (highest to lowest):
                       1. Values in this parameter
                       2. Values from radar-specific TOML section
                       3. Values from [defaults] TOML section
                       4. Package default radar_defaults.toml

                       Common fields:
                       - antenna_height_m: Antenna elevation above sea level (meters)
                       - lowest_level_offset_m: Lowest profile level above the
                         antenna (meters); override together with antenna_height_m
                       - beamwidth_deg: One-way half-power beamwidth (degrees)
                       - freezing_level_m: Freezing level in meters ASL (from NWP)
                         * None: Unknown (not yet retrieved from NWP data)
                         * 0 or negative: No freezing layer → normalized to 0
                         * Positive: Valid freezing level (converted to above-antenna internally)

    Returns:
        xarray.Dataset with profile data and metadata. Heights form a
        contiguous 200 m grid from ``lowest_level_offset_m`` above the antenna
        to the highest level in the file; levels absent from the file are
        no-echo levels (MDS, zero sample counts) listed in
        ``attrs['missing_levels_m']``.

    Raises:
        NoUsableProfileError: If the file is a header-only product or has no
            levels at or above the lowest profile level.
        ValueError: If the header radar name is not configured as a radar
            ``name`` or ``aliases`` entry, the radar geometry is incomplete or
            does not match the file, or the profile is invalid.

    Environment Variables:
        VPRC_RADAR_CONFIG: Path to custom radar configuration TOML file.
                          If not set, uses package default radar_defaults.toml.

    Example:
        >>> # Use all defaults from TOML
        >>> ds = read_vvp('202508241100_KAN.VVP_40.txt')
        >>> ds.attrs['antenna_height_m']
        174
        >>>
        >>> # Override freezing level from NWP data
        >>> ds = read_vvp('202508241100_KAN.VVP_40.txt',
        ...               {'freezing_level_m': 2000})
        >>> ds.attrs['freezing_level_m']
        1826
        >>>
        >>> # Use custom config file
        >>> import os
        >>> os.environ['VPRC_RADAR_CONFIG'] = '/path/to/custom_radars.toml'
        >>> ds = read_vvp('202508241100_KAN.VVP_40.txt')
    """
    filepath = Path(filepath)
    header, df = _parse_vvp_file(filepath)
    ds = _vvp_dataframe_to_xarray(df, header, radar_metadata, source_file=filepath)
    return ds


def height_to_sea_level(ds: xr.Dataset) -> xr.DataArray:
    """
    Convert height coordinate from above-antenna to sea level.

    The VPR correction uses heights above antenna for processing (since
    observation geometry is relative to antenna position). This helper
    converts back to sea level heights for output or comparison with
    other data sources.

    Args:
        ds: Dataset with height coordinate in meters above antenna
            and antenna_height_m in attrs

    Returns:
        DataArray with heights in meters above sea level

    Example:
        >>> ds = read_vvp('profile.txt')
        >>> ds.height.values[:3]  # Heights above antenna
        array([126, 326, 526])
        >>> height_to_sea_level(ds).values[:3]  # Sea level
        array([300, 500, 700])
    """
    antenna_height_m = ds.attrs.get("antenna_height_m", 0)
    return ds.height + antenna_height_m
