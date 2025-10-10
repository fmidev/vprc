#!/usr/bin/env python3
"""
Parser for IRIS VVP (Vertical Velocity Profile) prodx files.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import xarray as xr


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

    Args:
        df: DataFrame from parse_vvp_file
        header: VVPHeader metadata
        radar_metadata: Optional dict with antenna_height_km, lowest_level_m, etc.

    Returns:
        xarray.Dataset with dimensions (height,) and rich metadata

    Note:
        For multi-profile processing (multiple scans), you'd extend this
        to have dimensions (height, profile_idx).
    """
    # Convert DataFrame to xarray using height as the index/coordinate
    ds = df.set_index('height').to_xarray()

    # Add corrected_dbz as a copy of lin_dbz (will be modified by processing)
    ds['corrected_dbz'] = ds['lin_dbz'].copy()

    # Build metadata attributes
    attrs = {
        'radar': header.radar,
        'product': header.product,
        'scan_type': header.scan_type,
        'timestamp': header.timestamp.isoformat(),
        'elevation_angles': header.elevation_angles,
        'source_file': str(source_file) if source_file else 'unknown',
    }

    # Add radar-specific metadata if provided
    if radar_metadata:
        attrs.update(radar_metadata)

    ds.attrs.update(attrs)

    return ds


def read_vvp(filepath: Path | str,
             radar_metadata: dict | None = None) -> xr.Dataset:
    """
    Parse VVP file directly to xarray Dataset (high-level wrapper).

    This is the recommended API for simple use cases.

    Args:
        filepath: Path to VVP prodx file
        radar_metadata: Optional dict with antenna_height_km, lowest_level_m,
                       freezing_level_m, etc.

    Returns:
        Tuple of (VVPHeader, xarray.Dataset)

    Example:
        >>> header, ds = parse_vvp_to_xarray('202508241100_KAN.VVP_40.txt')
        >>> ds['corrected_dbz'].sel(height=1500)
        <xarray.DataArray ...>
    """
    filepath = Path(filepath)
    header, df = _parse_vvp_file(filepath)
    ds = _vvp_dataframe_to_xarray(df, header, radar_metadata, source_file=filepath)
    return ds

