#!/usr/bin/env python3
"""
Tests for VVP file parsing functionality.
"""

from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

from vprc.io import (
    VVPHeader,
    _parse_vvp_header,
    _parse_vvp_file,
    _vvp_dataframe_to_xarray,
    read_vvp,
)


# Test data path
TEST_DATA_DIR = Path(__file__).parent / "data"
SAMPLE_VVP_FILE = TEST_DATA_DIR / "202508241100_KAN.VVP_40.txt"


class TestParseVVPHeader:
    """Tests for parse_vvp_header function."""

    def test_parse_valid_header(self):
        """Test parsing a valid VVP header line."""
        header_line = "KANKAANPAA VVP_40 PPI1_A 2025 08 24 11 00 ELEVS:  0.7 1.5 3.0"
        header = _parse_vvp_header(header_line)

        assert header.radar == "KANKAANPAA"
        assert header.product == "VVP_40"
        assert header.scan_type == "PPI1_A"
        assert header.timestamp == datetime(2025, 8, 24, 11, 0)
        assert header.elevation_angles == [0.7, 1.5, 3.0]

    def test_parse_header_single_elevation(self):
        """Test parsing header with single elevation angle."""
        header_line = "RADAR1 VVP_10 SCAN_A 2025 01 15 12 30 ELEVS: 1.0"
        header = _parse_vvp_header(header_line)

        assert header.elevation_angles == [1.0]

    def test_parse_header_invalid_too_short(self):
        """Test that short invalid header raises ValueError."""
        header_line = "RADAR VVP_40 2025"
        with pytest.raises(ValueError, match="Invalid header"):
            _parse_vvp_header(header_line)


class TestParseVVPFile:
    """Tests for parse_vvp_file function."""

    def test_parse_sample_file(self):
        """Test parsing the sample VVP file."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        header, df = _parse_vvp_file(SAMPLE_VVP_FILE)

        # Check header
        assert header.radar == "KANKAANPAA"
        assert header.product == "VVP_40"
        assert header.timestamp == datetime(2025, 8, 24, 11, 0)
        assert header.elevation_angles == [0.7, 1.5, 3.0]

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 35  # Number of height levels in sample file
        assert df['height'].min() == 100
        assert df['height'].max() == 6900

    def test_dataframe_columns(self):
        """Test that DataFrame has expected columns."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        _, df = _parse_vvp_file(SAMPLE_VVP_FILE)

        expected_cols = [
            'height', 'count', 'zcount',
            'wind_speed', 'wind_speed_std',
            'direction', 'direction_std',
            'vertical', 'vertical_std',
            'lin_dbz', 'lin_dbz_std',
            'log_dbz', 'log_dbz_std',
            'divergence', 'divergence_std',
            'deformation', 'deformation_std',
            'axis_of_dil', 'axis_of_dil_std'
        ]

        assert list(df.columns) == expected_cols

    def test_dataframe_sorted_by_height(self):
        """Test that DataFrame is sorted by height ascending."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        _, df = _parse_vvp_file(SAMPLE_VVP_FILE)

        # Check that heights are monotonically increasing
        assert df['height'].is_monotonic_increasing

    def test_dataframe_height_dtype(self):
        """Test that height column is integer type."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        _, df = _parse_vvp_file(SAMPLE_VVP_FILE)

        assert df['height'].dtype == int

    def test_dataframe_values(self):
        """Test specific data values from sample file."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        _, df = _parse_vvp_file(SAMPLE_VVP_FILE)

        # Check first row (height=100 in sorted data)
        row_100 = df[df['height'] == 100].iloc[0]
        assert row_100['count'] == 1326
        assert row_100['zcount'] == 1326
        assert abs(row_100['wind_speed'] - 5.5) < 0.01
        assert abs(row_100['direction'] - 350.0) < 0.01

        # Check a middle row (height=1500)
        row_1500 = df[df['height'] == 1500].iloc[0]
        assert row_1500['count'] == 5000
        assert abs(row_1500['wind_speed'] - 7.5) < 0.01
        assert abs(row_1500['lin_dbz'] - 30.0) < 0.1

    def test_parse_nonexistent_file(self):
        """Test that parsing nonexistent file raises FileNotFoundError."""
        nonexistent = TEST_DATA_DIR / "nonexistent_file.txt"
        with pytest.raises(FileNotFoundError):
            _parse_vvp_file(nonexistent)

    def test_parse_file_too_short(self, tmp_path: Path):
        """Test that file with too few lines raises ValueError."""
        short_file = tmp_path / "short.txt"
        short_file.write_text("RADAR VVP_40 PPI1_A 2025 08 24 11 00\n\n")

        with pytest.raises(ValueError, match="File too short"):
            _parse_vvp_file(short_file)


class TestVVPDataframeToXarray:
    """Tests for vvp_dataframe_to_xarray function."""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample parsed data."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")
        return _parse_vvp_file(SAMPLE_VVP_FILE)

    def test_convert_to_xarray(self, sample_data: Tuple[VVPHeader | pd.DataFrame]):
        """Test conversion to xarray Dataset."""
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        # Check that it's an xarray Dataset
        import xarray as xr
        assert isinstance(ds, xr.Dataset)

    def test_xarray_dimensions(self, sample_data: Tuple[VVPHeader | pd.DataFrame]):
        """Test that xarray has correct dimensions."""
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        assert 'height' in ds.dims
        assert ds.dims['height'] == len(df)

    def test_xarray_coordinates(self, sample_data: Tuple[VVPHeader | pd.DataFrame]):
        """Test that xarray has correct coordinates."""
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        assert 'height' in ds.coords
        assert len(ds.coords['height']) == len(df)
        assert ds.coords['height'].min().values == 100
        assert ds.coords['height'].max().values == 6900

    def test_xarray_variables(self, sample_data: Tuple[VVPHeader | pd.DataFrame]):
        """Test that xarray has expected data variables."""
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        expected_vars = [
            'count', 'zcount',
            'wind_speed', 'wind_speed_std',
            'direction', 'direction_std',
            'vertical', 'vertical_std',
            'lin_dbz', 'lin_dbz_std',
            'log_dbz', 'log_dbz_std',
            'divergence', 'divergence_std',
            'deformation', 'deformation_std',
            'axis_of_dil', 'axis_of_dil_std',
            'corrected_dbz'  # Added by conversion function
        ]

        for var in expected_vars:
            assert var in ds.data_vars, f"Variable {var} not in dataset"

    def test_xarray_corrected_dbz(self, sample_data: Tuple[VVPHeader | pd.DataFrame]):
        """Test that corrected_dbz is initialized as copy of lin_dbz."""
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        # Should be equal initially
        assert (ds['corrected_dbz'] == ds['lin_dbz']).all()

    def test_xarray_metadata(self, sample_data: Tuple[VVPHeader | pd.DataFrame]):
        """Test that xarray has correct metadata attributes."""
        header, df = sample_data
        radar_meta = {
            'antenna_height_km': 0.174,
            'lowest_level_offset_m': 126,
        }
        ds = _vvp_dataframe_to_xarray(df, header, radar_meta, SAMPLE_VVP_FILE)

        assert ds.attrs['radar'] == 'KANKAANPAA'
        assert ds.attrs['product'] == 'VVP_40'
        assert ds.attrs['scan_type'] == 'PPI1_A'
        assert ds.attrs['timestamp'] == '2025-08-24T11:00:00'
        assert ds.attrs['elevation_angles'] == [0.7, 1.5, 3.0]
        assert ds.attrs['antenna_height_km'] == 0.174
        assert ds.attrs['lowest_level_offset_m'] == 126
        assert 'source_file' in ds.attrs

    def test_xarray_data_access(self, sample_data: Tuple[VVPHeader | pd.DataFrame]):
        """Test accessing data from xarray Dataset."""
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        # Test selection by height
        data_at_1500 = ds.sel(height=1500)
        assert data_at_1500['count'].values == 5000
        assert abs(float(data_at_1500['wind_speed'].values) - 7.5) < 0.01


class TestParseVVPToXarray:
    """Tests for parse_vvp_to_xarray high-level function."""

    def test_parse_to_xarray(self):
        """Test high-level parsing function."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        ds = read_vvp(SAMPLE_VVP_FILE)

        # Check dataset
        import xarray as xr
        assert isinstance(ds, xr.Dataset)
        assert 'height' in ds.dims
        assert 'corrected_dbz' in ds.data_vars

    def test_parse_to_xarray_with_metadata(self):
        """Test parsing with custom radar metadata."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        radar_meta = {
            'antenna_height_km': 0.174,
            'lowest_level_offset_m': 126,
            'freezing_level_m': 2000,
        }
        ds = read_vvp(SAMPLE_VVP_FILE, radar_meta)

        assert ds.attrs['antenna_height_km'] == 0.174
        assert ds.attrs['freezing_level_m'] == 2000

    def test_roundtrip_consistency(self):
        """Test that step-by-step and high-level APIs produce same results."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        # Method 1: Step by step
        header1, df1 = _parse_vvp_file(SAMPLE_VVP_FILE)
        ds1 = _vvp_dataframe_to_xarray(df1, header1)

        # Method 2: High-level wrapper
        ds2 = read_vvp(SAMPLE_VVP_FILE)

        # Data should be identical
        assert (ds1['lin_dbz'] == ds2['lin_dbz']).all()
        assert (ds1['wind_speed'] == ds2['wind_speed']).all()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe_handling(self):
        """Test handling of edge case with minimal valid data."""
        # This tests behavior when file structure is valid but data is minimal
        # Implementation depends on how you want to handle such cases
        pass

    def test_pathlib_path_input(self):
        """Test that both Path and str inputs work."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        # Test with Path object
        header1, df1 = _parse_vvp_file(SAMPLE_VVP_FILE)

        # Test with string
        header2, df2 = _parse_vvp_file(str(SAMPLE_VVP_FILE))

        assert header1.radar == header2.radar
        assert len(df1) == len(df2)
