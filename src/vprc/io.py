#!/usr/bin/env python3
"""
Parser for IRIS VVP (Vertical Velocity Profile) prodx files.
"""

import os
import tomllib
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import xarray as xr


@lru_cache(maxsize=1)
def _load_radar_defaults() -> dict:
    """
    Load radar default configurations from TOML file.

    Configuration precedence:
    1. TOML file specified by VPRC_RADAR_CONFIG environment variable
    2. Package-shipped radar_defaults.toml

    Returns:
        Dictionary with radar codes as keys, each containing
        antenna_height_m and lowest_level_offset_m.

    Note:
        Results are cached. Only loaded once per process.
    """
    # Check for user-specified config file
    config_path = os.environ.get('VPRC_RADAR_CONFIG')

    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(
                f"VPRC_RADAR_CONFIG points to non-existent file: {config_path}"
            )
    else:
        # Use package default
        config_file = Path(__file__).parent / 'radar_defaults.toml'

    with open(config_file, 'rb') as f:
        return tomllib.load(f)


@lru_cache(maxsize=1)
def _build_radar_name_to_code_map() -> dict[str, str]:
    """
    Build reverse mapping from full radar names to three-letter codes.

    This mapping is derived from the TOML configuration file, avoiding
    hardcoded name-to-code relationships in Python code.

    Returns:
        Dictionary mapping full radar names (uppercase) to codes.
        Example: {"KANKAANPAA": "KAN", "VANTAA": "VAN", ...}

    Note:
        Results are cached. Only built once per process.
        Reuses the cached TOML data from _load_radar_defaults().
    """
    radar_defaults = _load_radar_defaults()
    name_to_code = {}

    for code, config in radar_defaults.items():
        if code == 'other':
            continue
        # Get the 'name' field from config
        if 'name' in config:
            name_to_code[config['name'].upper()] = code

    return name_to_code


def _radar_name_to_code(radar_name: str) -> str:
    """
    Convert full radar name to three-letter code.

    Args:
        radar_name: Full radar name from VVP file header (e.g., "KANKAANPAA")

    Returns:
        Three-letter radar code (e.g., "KAN")
        If not found in mapping, returns the radar_name unchanged
        (will fallback to 'other' in _get_radar_metadata).

    Example:
        >>> _radar_name_to_code("KANKAANPAA")
        'KAN'
        >>> _radar_name_to_code("UNKNOWN_RADAR")
        'UNKNOWN_RADAR'
    """
    name_map = _build_radar_name_to_code_map()
    return name_map.get(radar_name.upper(), radar_name)


def _get_radar_metadata(radar_code: str,
                       override_metadata: dict | None = None) -> dict:
    """
    Get radar metadata with proper precedence.

    Precedence (highest to lowest):
    1. Values provided in override_metadata parameter
    2. Values from radar-specific section in TOML
    3. Values from [defaults] section in TOML (used as fallback for unknown radars)

    Args:
        radar_code: Three-letter radar station code (e.g., 'KAN', 'VAN')
        override_metadata: Optional dict to override specific fields

    Returns:
        Dictionary with antenna_height_m, lowest_level_offset_m, beamwidth_deg,
        and any additional fields from override_metadata (e.g., freezing_level_m)

    Example:
        >>> meta = _get_radar_metadata('KAN', {'freezing_level_m': 2000})
        >>> meta['antenna_height_m']
        174
        >>> meta['freezing_level_m']
        2000
        >>> meta['beamwidth_deg']  # From [defaults] section
        0.95
    """
    # Load defaults from TOML
    radar_defaults = _load_radar_defaults()

    # Start with shared defaults (also serves as fallback for unknown radars)
    metadata = dict(radar_defaults.get('defaults', {}))

    # Get radar-specific config if it exists
    if radar_code in radar_defaults and radar_code != 'defaults':
        radar_config = radar_defaults[radar_code]
        # Update with radar-specific config (overrides defaults)
        metadata.update(radar_config)

    # Override with user-supplied values (highest priority)
    if override_metadata:
        metadata.update(override_metadata)

    return metadata


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
        raise ValueError(f"Invalid header: expected at least 9 fields, got {len(parts)}")

    radar = parts[0]
    product = parts[1]
    scan_type = parts[2]
    year = int(parts[3])
    month = int(parts[4])
    day = int(parts[5])
    hour = int(parts[6])
    minute = int(parts[7])

    # Find ELEVS: keyword and extract elevation angles
    try:
        elevs_idx = parts.index('ELEVS:')
        elevation_angles = [float(x) for x in parts[elevs_idx + 1:]]
    except (ValueError, IndexError):
        elevation_angles = []

    timestamp = datetime(year, month, day, hour, minute)

    return VVPHeader(
        radar=radar,
        product=product,
        scan_type=scan_type,
        timestamp=timestamp,
        elevation_angles=elevation_angles
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

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise ValueError(f"File too short: expected at least 4 lines, got {len(lines)}")

    # Parse header (line 0)
    header = _parse_vvp_header(lines[0])

    # The file has a multi-level header structure where each major column
    # (Wind-Speed, Linear-dBZ, etc.) has 2 sub-values (mean and std deviation)
    # We'll use read_fwf with colspecs or just treat it as whitespace-delimited
    # and manually assign proper column names

    # Strategy: Skip header lines, read all numeric columns, then assign names
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        skiprows=3,          # Skip header, blank, and column name line
        header=None,         # No header in data
        engine='python'
    )

    # Assign column names based on the structure:
    # Height Count ZCnt  Wind-Speed(2) Direction(2) Vertical(2) Linear-dBZ(2) Log-dBZ(2)
    # Divergence(2) Deformation(2) Axis-of-Dil(2)
    # Total: 1 + 2 + 2*10 = 23 columns

    expected_cols = [
        'height',
        'count', 'zcount',
        'wind_speed', 'wind_speed_std',
        'direction', 'direction_std',
        'vertical', 'vertical_std',
        'lin_dbz', 'lin_dbz_std',
        'log_dbz', 'log_dbz_std',
        'divergence', 'divergence_std',
        'deformation', 'deformation_std',
        'axis_of_dil', 'axis_of_dil_std'
    ]

    # Verify column count
    if len(df.columns) != len(expected_cols):
        # Fallback: use generic names
        print(f"Warning: Expected {len(expected_cols)} columns, got {len(df.columns)}")
        df.columns = [f'col_{i}' for i in range(len(df.columns))]
    else:
        df.columns = expected_cols

    # Ensure height is integer
    df['height'] = df['height'].astype(int)

    # Sort by height ascending (files go high->low)
    df = df.sort_values('height').reset_index(drop=True)

    return header, df


def _vvp_dataframe_to_xarray(df: pd.DataFrame, header: VVPHeader,
                             radar_metadata: dict | None = None,
                             source_file: Path | str | None = None) -> xr.Dataset:
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
                       - lowest_level_offset_m: Offset from nearest profile level
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
    # Resolve metadata with proper precedence
    # Extract radar code from header using TOML-based mapping
    radar_name = header.radar.upper()
    radar_code = _radar_name_to_code(radar_name)

    # Get merged metadata (TOML defaults + overrides)
    metadata = _get_radar_metadata(radar_code, radar_metadata)

    # Convert sea-level heights to heights above antenna
    # This follows the legacy Perl logic: $Height = $SeaHeight - $antennin_korkeus * 1000
    # (allprof_prodx2.pl line 153)
    antenna_height_m = metadata.get('antenna_height_m', 0)
    df = df.copy()
    df['height'] = df['height'] - antenna_height_m

    # Drop levels below antenna (negative heights)
    # Legacy: if ($Height < $alintaso) { $Height = 0; } but actually skips them
    df = df[df['height'] > 0].copy()

    # Convert DataFrame to xarray using height as the index/coordinate
    ds = df.set_index('height').to_xarray()

    # Rename 'count' to 'sample_count' for consistency with processing modules
    if 'count' in ds:
        ds = ds.rename({'count': 'sample_count'})

    # Add corrected_dbz as a copy of lin_dbz (will be modified by processing)
    ds['corrected_dbz'] = ds['lin_dbz'].copy()

    # Build metadata attributes
    attrs = {
        'radar': header.radar,
        'radar_code': radar_code,
        'product': header.product,
        'scan_type': header.scan_type,
        'timestamp': header.timestamp.isoformat(),
        'elevation_angles': header.elevation_angles,
        'source_file': str(source_file) if source_file else 'unknown',
    }

    # Add radar-specific metadata
    # Handle freezing_level_m specially with three cases:
    # 1. None or not provided: Unknown (not yet retrieved from NWP) → keep as None
    # 2. Zero or negative (ASL or above-antenna): No melting layer visible → set to 0
    # 3. Positive above-antenna: Valid freezing level for BB detection
    # (following Perl logic from allprof_prodx2.pl lines 359-362)
    metadata_copy = metadata.copy()
    if 'freezing_level_m' in metadata_copy:
        fl = metadata_copy['freezing_level_m']
        if fl is not None and isinstance(fl, (int, float)):
            if fl <= 0:
                # Explicitly no freezing layer (or below surface)
                metadata_copy['freezing_level_m'] = 0
            else:
                # Convert from ASL to above-antenna (same as height coordinate)
                # If result is ≤0, freezing level is at/below antenna → no melting visible
                fl_above_antenna = fl - antenna_height_m
                metadata_copy['freezing_level_m'] = max(0, fl_above_antenna)
        # else: None stays None (unknown)

    attrs.update(metadata_copy)

    ds.attrs.update(attrs)

    return ds


def read_vvp(filepath: Path | str,
             radar_metadata: dict | None = None) -> xr.Dataset:
    """
    Parse VVP file directly to xarray Dataset (high-level wrapper).

    This is the recommended API for simple use cases.

    Args:
        filepath: Path to VVP prodx file
        radar_metadata: Optional dict to override specific metadata fields.
                       Configuration precedence (highest to lowest):
                       1. Values in this parameter
                       2. Values from radar-specific TOML section
                       3. Values from [defaults] TOML section (fallback)
                       4. Package default radar_defaults.toml

                       Common fields:
                       - antenna_height_m: Antenna elevation above sea level (meters)
                       - lowest_level_offset_m: Offset from nearest profile level
                       - beamwidth_deg: One-way half-power beamwidth (degrees)
                       - freezing_level_m: Freezing level in meters ASL (from NWP)
                         * None: Unknown (not yet retrieved from NWP data)
                         * 0 or negative: No freezing layer → normalized to 0
                         * Positive: Valid freezing level (converted to above-antenna internally)

    Returns:
        xarray.Dataset with profile data and metadata

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
        2000
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
    antenna_height_m = ds.attrs.get('antenna_height_m', 0)
    return ds.height + antenna_height_m
