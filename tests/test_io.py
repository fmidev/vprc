#!/usr/bin/env python3
"""
Tests for VVP file parsing functionality.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from vprc.io import (
    VVPHeader,
    _parse_vvp_header,
    _parse_vvp_file,
    _vvp_dataframe_to_xarray,
    _load_radar_defaults,
    _build_radar_name_to_code_map,
    _radar_name_to_code,
    _get_radar_metadata,
    read_vvp,
)
from vprc.config import list_radar_codes, resolve_radar_code
from vprc.constants import MDS

# Test data path
TEST_DATA_DIR = Path(__file__).parent / "data"
SAMPLE_VVP_FILE = TEST_DATA_DIR / "202508241100_KAN.VVP_40.txt"
KOR_VVP_FILE = TEST_DATA_DIR / "202308281000_KOR.VVP_40.txt"
# Operational KESALAHTI profile that lacks the 500, 1300 and 1700 m ASL levels.
GAPPED_VVP_FILE = TEST_DATA_DIR / "202611112000_KES.VVP_40.txt"


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

    @pytest.mark.parametrize("angles", ["0.7 invalid", "nan", "inf"])
    def test_invalid_elevations_raise(self, angles):
        with pytest.raises(ValueError):
            _parse_vvp_header(
                f"KANKAANPAA VVP_40 PPI1_A 2025 08 24 11 00 ELEVS: {angles}"
            )


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
        """Test that xarray has correct dimensions.

        Note: heights are converted to above-antenna coordinates, and levels
        below antenna are dropped. For KAN (antenna at 174m), input level 100m
        is dropped (100-174 < 0), leaving 34 of original 35 levels.
        """
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        assert 'height' in ds.dims
        # Original 35 levels minus 1 dropped (below antenna)
        assert ds.sizes['height'] == len(df) - 1

    def test_xarray_coordinates(self, sample_data: Tuple[VVPHeader | pd.DataFrame]):
        """Test that xarray has correct coordinates.

        Heights are converted to above-antenna: sea_level - antenna_height_m.
        For KAN (antenna 174m): 300m ASL -> 126m above antenna,
        6900m ASL -> 6726m above antenna.
        """
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        assert 'height' in ds.coords
        # 35 input levels, 1 dropped (below antenna)
        assert len(ds.coords['height']) == len(df) - 1
        # Lowest remaining level: 300m ASL - 174m = 126m above antenna
        assert ds.coords['height'].min().values == 126
        # Highest level: 6900m ASL - 174m = 6726m above antenna
        assert ds.coords['height'].max().values == 6726

    def test_xarray_variables(self, sample_data: Tuple[VVPHeader | pd.DataFrame]):
        """Test that xarray has expected data variables."""
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        expected_vars = [
            'sample_count', 'zcount',
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
        """Test accessing data from xarray Dataset.

        Heights are above antenna, so 1500m ASL -> 1326m above antenna.
        """
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        # Test selection by height (above antenna)
        # Original 1500m ASL - 174m antenna = 1326m above antenna
        data_at_1326 = ds.sel(height=1326)
        assert data_at_1326['sample_count'].values == 5000
        assert abs(float(data_at_1326['wind_speed'].values) - 7.5) < 0.01


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

        # KAN antenna height is 174m
        antenna_height = 174
        radar_meta = {
            'antenna_height_km': 0.174,
            'lowest_level_offset_m': 126,
            'freezing_level_m': 2000,  # ASL
        }
        ds = read_vvp(SAMPLE_VVP_FILE, radar_meta)

        assert ds.attrs['antenna_height_km'] == 0.174
        # freezing_level converted from ASL (2000) to above-antenna (2000 - 174)
        assert ds.attrs['freezing_level_m'] == 2000 - antenna_height

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

    def test_no_levels_above_antenna_raises(self):
        header, df = _parse_vvp_file(SAMPLE_VVP_FILE)
        with pytest.raises(ValueError, match="no levels above"):
            _vvp_dataframe_to_xarray(df.iloc[:0], header)

    @pytest.mark.parametrize("freezing_level", [float("nan"), float("inf"), "2000", True])
    def test_invalid_freezing_level_raises(self, freezing_level):
        with pytest.raises(ValueError, match="freezing_level_m must be finite"):
            read_vvp(SAMPLE_VVP_FILE, {"freezing_level_m": freezing_level})

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            ("100 30 40", "Expected 19 columns"),
            ("100 30 40 0 0 0 0 0 0 nan 0 0 0 0 0 0 0 0 0", "lin_dbz"),
            ("100.5 30 40 0 0 0 0 0 0 10 0 0 0 0 0 0 0 0 0", "integer meters"),
        ],
    )
    def test_corrupt_profile_data_raises(self, tmp_path, row, message):
        path = tmp_path / "invalid.txt"
        path.write_text(
            "KANKAANPAA VVP_40 PPI1_A 2025 08 24 11 00 ELEVS: 0.7\n\n"
            "Height Count ZCnt ...\n" + row + "\n"
        )
        with pytest.raises(ValueError, match=message):
            read_vvp(path)

    def test_reflectivity_count_is_not_wind_count(self, tmp_path):
        path = tmp_path / "counts.txt"
        path.write_text(
            "KANKAANPAA VVP_40 PPI1_A 2025 08 24 11 00 ELEVS: 0.7\n\n"
            "Height Count ZCnt ...\n"
            "300 0 1000 0 0 0 0 0 0 20 0 0 0 0 0 0 0 0 0\n"
        )
        ds = read_vvp(path)
        assert ds.sample_count.item() == 0
        assert ds.zcount.item() == 1000

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

    def test_missing_freezing_level(self):
        """Test handling when freezing_level_m is not provided or invalid.

        Three cases:
        1. Unknown (None/not provided): Keep as None
        2. No freezing layer (0 or negative): Normalize to 0
        3. Valid (positive ASL): Convert to above-antenna
        """
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        # KAN antenna height is 174m
        antenna_height = 174

        # Case 1: Unknown - No metadata provided
        ds1 = read_vvp(SAMPLE_VVP_FILE, radar_metadata={'antenna_height_km': 0.174})
        assert ds1.attrs.get('freezing_level_m') is None

        # Case 1: Unknown - freezing_level_m is None
        ds2 = read_vvp(SAMPLE_VVP_FILE, radar_metadata={'freezing_level_m': None})
        assert ds2.attrs.get('freezing_level_m') is None

        # Case 2: No freezing layer - negative (below surface)
        ds3 = read_vvp(SAMPLE_VVP_FILE, radar_metadata={'freezing_level_m': -100})
        assert ds3.attrs.get('freezing_level_m') == 0

        # Case 2: No freezing layer - zero (at sea level)
        ds4 = read_vvp(SAMPLE_VVP_FILE, radar_metadata={'freezing_level_m': 0})
        assert ds4.attrs.get('freezing_level_m') == 0

        # Case 2b: Freezing level positive ASL but below antenna (e.g., 100m ASL < 174m antenna)
        # No melting layer visible to radar → normalized to 0
        ds4b = read_vvp(SAMPLE_VVP_FILE, radar_metadata={'freezing_level_m': 100})
        assert ds4b.attrs.get('freezing_level_m') == 0

        # Case 3: Valid freezing layer - positive ASL value
        # Input 2000m ASL -> stored as 2000 - 174 = 1826m above antenna
        ds5 = read_vvp(SAMPLE_VVP_FILE, radar_metadata={'freezing_level_m': 2000})
        assert ds5.attrs['freezing_level_m'] == 2000 - antenna_height


class TestRadarConfiguration:
    """Tests for radar configuration loading and precedence."""

    def test_load_radar_defaults(self):
        """Test loading the package default radar configuration."""
        config = _load_radar_defaults()

        # Check that all expected radar codes are present
        assert 'KAN' in config
        assert 'VAN' in config
        assert 'defaults' in config

        # Check structure of a known radar
        assert 'antenna_height_m' in config['KAN']
        assert 'lowest_level_offset_m' in config['KAN']

        # Check structure of defaults section
        assert 'antenna_height_m' in config['defaults']
        assert 'lowest_level_offset_m' in config['defaults']
        assert 'beamwidth_deg' in config['defaults']

        # Verify known values from Perl reference
        assert config['KAN']['antenna_height_m'] == 174
        assert config['KAN']['lowest_level_offset_m'] == 126

    def test_get_radar_metadata_known_radar(self):
        """Test retrieving metadata for a known radar code."""
        meta = _get_radar_metadata('KAN')

        assert meta['antenna_height_m'] == 174
        assert meta['lowest_level_offset_m'] == 126

    def test_get_radar_metadata_unknown_radar(self):
        """Test fallback to [defaults] for unknown radar code."""
        meta = _get_radar_metadata('UNKNOWN')

        # Should get [defaults] values
        assert meta['antenna_height_m'] == 198
        assert meta['lowest_level_offset_m'] == 102
        assert meta['beamwidth_deg'] == 0.95

    def test_get_radar_metadata_with_override(self):
        """Test that override metadata takes precedence."""
        meta = _get_radar_metadata('KAN', {'antenna_height_m': 999})

        # Overridden field
        assert meta['antenna_height_m'] == 999
        # Original field from TOML
        assert meta['lowest_level_offset_m'] == 126

    def test_get_radar_metadata_with_additional_fields(self):
        """Test adding additional fields via override."""
        meta = _get_radar_metadata('KAN', {'freezing_level_m': 2000})

        # Original fields from TOML
        assert meta['antenna_height_m'] == 174
        assert meta['lowest_level_offset_m'] == 126
        # New field from override
        assert meta['freezing_level_m'] == 2000

    def test_get_radar_metadata_includes_beamwidth(self):
        """Test that beamwidth_deg is loaded from [defaults] section."""
        meta = _get_radar_metadata('KAN')

        # Should include beamwidth from [defaults] section
        assert 'beamwidth_deg' in meta
        assert meta['beamwidth_deg'] == 0.95

    def test_get_radar_metadata_beamwidth_can_be_overridden(self):
        """Test that beamwidth_deg can be overridden per-radar."""
        meta = _get_radar_metadata('KAN', {'beamwidth_deg': 1.0})

        assert meta['beamwidth_deg'] == 1.0

    def test_list_radar_codes_returns_enabled_by_default(self):
        """Test that list_radar_codes returns enabled radars by default."""
        codes = list_radar_codes()

        # All radars in default config are enabled
        assert 'KAN' in codes
        assert 'VAN' in codes
        assert 'VIH' in codes
        # Should not include special sections
        assert 'defaults' not in codes
        assert 'network' not in codes

    def test_list_radar_codes_enabled_only_false(self):
        """Test that list_radar_codes(enabled_only=False) returns all radars."""
        codes_enabled = list_radar_codes(enabled_only=True)
        codes_all = list_radar_codes(enabled_only=False)

        # With default config, both should be equal
        assert set(codes_enabled) == set(codes_all)

    def test_list_radar_codes_respects_enabled_flag(self, tmp_path):
        """Test that disabled radars are excluded when enabled_only=True."""
        # Create custom config with one disabled radar
        custom_config = tmp_path / "test_enabled.toml"
        custom_config.write_text("""
[defaults]
enabled = true
antenna_height_m = 100
lowest_level_offset_m = 50

[KAN]
antenna_height_m = 174
enabled = true

[VAN]
antenna_height_m = 83
enabled = false

[VIH]
antenna_height_m = 181
# enabled not specified, should inherit from defaults (true)
""")

        old_env = os.environ.get('VPRC_RADAR_CONFIG')
        try:
            os.environ['VPRC_RADAR_CONFIG'] = str(custom_config)
            _load_radar_defaults.cache_clear()
            _build_radar_name_to_code_map.cache_clear()

            # enabled_only=True (default) should exclude VAN
            codes_enabled = list_radar_codes(enabled_only=True)
            assert 'KAN' in codes_enabled
            assert 'VIH' in codes_enabled
            assert 'VAN' not in codes_enabled

            # enabled_only=False should include all
            codes_all = list_radar_codes(enabled_only=False)
            assert 'KAN' in codes_all
            assert 'VIH' in codes_all
            assert 'VAN' in codes_all

        finally:
            if old_env is not None:
                os.environ['VPRC_RADAR_CONFIG'] = old_env
            else:
                os.environ.pop('VPRC_RADAR_CONFIG', None)
            _load_radar_defaults.cache_clear()
            _build_radar_name_to_code_map.cache_clear()

    def test_read_vvp_with_defaults(self):
        """Test that read_vvp loads TOML defaults automatically."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        ds = read_vvp(SAMPLE_VVP_FILE)

        # Should have loaded KAN radar defaults
        assert ds.attrs['radar_code'] == 'KAN'
        assert ds.attrs['antenna_height_m'] == 174
        assert ds.attrs['lowest_level_offset_m'] == 126
        # Should also include beamwidth from [defaults] section
        assert ds.attrs['beamwidth_deg'] == 0.95

    def test_read_vvp_with_override(self):
        """Test precedence: function parameter > TOML defaults."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        # Override antenna_height to 200m, freezing_level to 2500m ASL
        ds = read_vvp(SAMPLE_VVP_FILE, {'antenna_height_m': 200, 'freezing_level_m': 2500})

        # Overridden value
        assert ds.attrs['antenna_height_m'] == 200
        # TOML default
        assert ds.attrs['lowest_level_offset_m'] == 126
        # freezing_level converted from ASL (2500) to above-antenna (2500 - 200 = 2300)
        assert ds.attrs['freezing_level_m'] == 2300

    def test_custom_config_via_env_var(self, tmp_path):
        """Test loading custom TOML via VPRC_RADAR_CONFIG environment variable."""
        # Create a custom config file
        custom_config = tmp_path / "custom_radars.toml"
        custom_config.write_text("""
[KAN]
antenna_height_m = 999
lowest_level_offset_m = 888

[other]
antenna_height_m = 100
lowest_level_offset_m = 50
""")

        # Set environment variable and clear cache
        old_env = os.environ.get('VPRC_RADAR_CONFIG')
        try:
            os.environ['VPRC_RADAR_CONFIG'] = str(custom_config)
            # Clear the cache to force reload
            _load_radar_defaults.cache_clear()

            # Test that custom config is loaded
            config = _load_radar_defaults()
            assert config['KAN']['antenna_height_m'] == 999
            assert config['KAN']['lowest_level_offset_m'] == 888

        finally:
            # Restore environment
            if old_env is not None:
                os.environ['VPRC_RADAR_CONFIG'] = old_env
            else:
                os.environ.pop('VPRC_RADAR_CONFIG', None)
            # Clear cache again to restore default
            _load_radar_defaults.cache_clear()

    def test_precedence_order(self, tmp_path):
        """Test full precedence: function param > env var TOML > package default."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        # Create custom config
        custom_config = tmp_path / "test_radars.toml"
        custom_config.write_text("""
[KAN]
antenna_height_m = 500
lowest_level_offset_m = 400
custom_field = 123
""")

        old_env = os.environ.get('VPRC_RADAR_CONFIG')
        try:
            os.environ['VPRC_RADAR_CONFIG'] = str(custom_config)
            _load_radar_defaults.cache_clear()

            # Load with function override
            ds = read_vvp(SAMPLE_VVP_FILE, {'antenna_height_m': 300})

            # Function parameter wins
            assert ds.attrs['antenna_height_m'] == 300
            # Custom TOML provides these
            assert ds.attrs['lowest_level_offset_m'] == 400
            assert ds.attrs['custom_field'] == 123

        finally:
            if old_env is not None:
                os.environ['VPRC_RADAR_CONFIG'] = old_env
            else:
                os.environ.pop('VPRC_RADAR_CONFIG', None)
            _load_radar_defaults.cache_clear()


class TestRadarNameMapping:
    """Tests for radar name to code mapping from TOML configuration."""

    def test_build_radar_name_to_code_map(self):
        """Test building the name-to-code mapping from TOML."""
        name_map = _build_radar_name_to_code_map()

        # Check that all known radars are present
        assert 'KANKAANPAA' in name_map
        assert 'VANTAA' in name_map
        assert 'LUOSTO' in name_map

        # Verify correct mappings based on Perl reference
        assert name_map['KANKAANPAA'] == 'KAN'
        assert name_map['VANTAA'] == 'VAN'
        assert name_map['IKAALINEN'] == 'IKA'
        assert name_map['KESALAHTI'] == 'KES'
        assert name_map['PETAJAVESI'] == 'PET'
        assert name_map['ANJALANKOSKI'] == 'ANJ'
        assert name_map['KUOPIO'] == 'KUO'
        assert name_map['UTAJARVI'] == 'UTA'
        assert name_map['LUOSTO'] == 'LUO'
        assert name_map['KAUNISPAA'] == 'KAU'
        assert name_map['VIMPELI'] == 'VIM'
        assert name_map['NURMES'] == 'NUR'
        assert name_map['VIHTI'] == 'VIH'
        assert name_map["KORPPOO"] == "KOR"
        assert name_map["KOR"] == "KOR"  # Alias: older headers, Perl match
        assert name_map['KERAVA'] == 'KER'

        # 15 radars, excluding 'other'; Korppoo has one alias
        assert len(set(name_map.values())) == 15
        assert len(name_map) == 16

    def test_build_radar_name_to_code_map_excludes_other(self):
        """Test that 'other' section is not included in name mapping."""
        name_map = _build_radar_name_to_code_map()
        assert 'other' not in name_map.values()

    def test_radar_name_to_code_known_radar(self):
        """Test converting known radar names to codes."""
        assert _radar_name_to_code('KANKAANPAA') == 'KAN'
        assert _radar_name_to_code('VANTAA') == 'VAN'
        assert _radar_name_to_code('LUOSTO') == 'LUO'

    def test_radar_name_to_code_case_insensitive(self):
        """Test that name matching is case-insensitive."""
        assert _radar_name_to_code('kankaanpaa') == 'KAN'
        assert _radar_name_to_code('Kankaanpaa') == 'KAN'
        assert _radar_name_to_code('KANKAANPAA') == 'KAN'

    def test_radar_name_to_code_unknown_radar(self):
        """Test that unknown radar names are returned unchanged."""
        unknown_name = 'UNKNOWN_RADAR'
        assert _radar_name_to_code(unknown_name) == unknown_name

    def test_radar_name_to_code_with_custom_toml(self, tmp_path):
        """Test name mapping with custom TOML configuration."""
        # Create custom config with new radar
        custom_config = tmp_path / "custom_radars.toml"
        custom_config.write_text("""
[TST]
name = "TESTDAR"
antenna_height_m = 100
lowest_level_offset_m = 50

[other]
antenna_height_m = 200
lowest_level_offset_m = 100
""")

        old_env = os.environ.get('VPRC_RADAR_CONFIG')
        try:
            os.environ['VPRC_RADAR_CONFIG'] = str(custom_config)
            _load_radar_defaults.cache_clear()
            _build_radar_name_to_code_map.cache_clear()

            # Test that custom radar is in mapping
            assert _radar_name_to_code('TESTDAR') == 'TST'

        finally:
            if old_env is not None:
                os.environ['VPRC_RADAR_CONFIG'] = old_env
            else:
                os.environ.pop('VPRC_RADAR_CONFIG', None)
            _load_radar_defaults.cache_clear()
            _build_radar_name_to_code_map.cache_clear()

    def test_read_vvp_uses_name_mapping(self):
        """Test that read_vvp correctly uses TOML-based name mapping."""
        if not SAMPLE_VVP_FILE.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_VVP_FILE}")

        ds = read_vvp(SAMPLE_VVP_FILE)

        # File contains "KANKAANPAA", should map to "KAN"
        assert ds.attrs['radar'] == 'KANKAANPAA'
        assert ds.attrs['radar_code'] == 'KAN'

        # And should load KAN's configuration
        assert ds.attrs['antenna_height_m'] == 174
        assert ds.attrs['lowest_level_offset_m'] == 126

    def test_name_field_required_in_toml(self):
        """Test that TOML entries without 'name' field are skipped in mapping."""
        # This tests the robustness of the mapping builder
        name_map = _build_radar_name_to_code_map()

        # 'other' doesn't have a 'name' field and should not be in mapping
        # (checked implicitly - if it was there, it would fail with KeyError)
        assert len(set(name_map.values())) == 15  # Only the 15 named radars

    def test_toml_name_matches_perl_reference(self):
        """Test that TOML names exactly match Perl reference implementation."""
        # Based on allprof_prodx2.pl lines 75-135
        expected_mappings = {
            'VANTAA': 'VAN',
            'KERAVA': 'KER',
            'IKAALINEN': 'IKA',
            'KANKAANPAA': 'KAN',
            'KESALAHTI': 'KES',
            'PETAJAVESI': 'PET',
            'ANJALANKOSKI': 'ANJ',
            'KUOPIO': 'KUO',
            'KOR': 'KOR',
            'UTAJARVI': 'UTA',
            'LUOSTO': 'LUO',
            'KAUNISPAA': 'KAU',
            'VIMPELI': 'VIM',
            'NURMES': 'NUR',
            'VIHTI': 'VIH',
        }

        name_map = _build_radar_name_to_code_map()

        for name, expected_code in expected_mappings.items():
            assert name in name_map, f"Missing radar name: {name}"
            assert name_map[name] == expected_code, \
                f"Wrong code for {name}: expected {expected_code}, got {name_map[name]}"

    # Header names of the operational radars, verified against VVP files
    # in November 2026.
    @pytest.mark.parametrize(
        ("header_name", "code"),
        [
            ("ANJALANKOSKI", "ANJ"),
            ("KANKAANPAA", "KAN"),
            ("KAUNISPAA", "KAU"),
            ("KESALAHTI", "KES"),
            ("KORPPOO", "KOR"),
            ("KUOPIO", "KUO"),
            ("LUOSTO", "LUO"),
            ("NURMES", "NUR"),
            ("PETAJAVESI", "PET"),
            ("UTAJARVI", "UTA"),
            ("VIHTI", "VIH"),
            ("VIMPELI", "VIM"),
        ],
    )
    def test_operational_header_names_resolve(self, header_name, code):
        assert resolve_radar_code(header_name) == code

    def test_korppoo_header_uses_kor_metadata(self, tmp_path):
        """Regression: KORPPOO headers fell back to [defaults] (antenna 198 m)."""
        lines = KOR_VVP_FILE.read_text().splitlines()
        lines[0] = "KORPPOO VVP_40 PPI1_A 2026 11 01 22 00 ELEVS:  0.7 1.5 3.0"
        path = tmp_path / "202611012200_KOR.VVP_40.txt"
        path.write_text("\n".join(lines) + "\n")

        ds = read_vvp(path)

        assert ds.attrs["radar"] == "KORPPOO"
        assert ds.attrs["radar_code"] == "KOR"
        assert ds.attrs["antenna_height_m"] == 61
        assert ds.attrs["lowest_level_offset_m"] == 39
        assert ds.attrs["latitude"] == pytest.approx(60.128469)
        assert "aliases" not in ds.attrs
        # The 100 m ASL level is kept at 100 - 61 m above the antenna.
        assert int(ds.height[0]) == 39

    def test_legacy_kor_header_still_resolves(self):
        ds = read_vvp(KOR_VVP_FILE)
        assert ds.attrs["radar"] == "KOR"
        assert ds.attrs["radar_code"] == "KOR"
        assert ds.attrs["antenna_height_m"] == 61

    def test_unconfigured_header_name_raises(self, tmp_path):
        lines = SAMPLE_VVP_FILE.read_text().splitlines()
        lines[0] = lines[0].replace("KANKAANPAA", "NEWRADAR")
        path = tmp_path / "unknown.txt"
        path.write_text("\n".join(lines) + "\n")

        with pytest.raises(ValueError, match="'NEWRADAR'.*not configured"):
            read_vvp(path)

    def test_header_name_claimed_by_two_radars_raises(self, tmp_path, monkeypatch):
        config = tmp_path / "duplicate.toml"
        config.write_text(
            '[AAA]\nname = "FIRST"\naliases = ["SHARED"]\n[BBB]\nname = "SHARED"\n'
        )
        monkeypatch.setenv("VPRC_RADAR_CONFIG", str(config))
        _load_radar_defaults.cache_clear()
        _build_radar_name_to_code_map.cache_clear()
        try:
            with pytest.raises(ValueError, match="'SHARED'.*AAA and BBB"):
                _build_radar_name_to_code_map()
        finally:
            monkeypatch.delenv("VPRC_RADAR_CONFIG")
            _load_radar_defaults.cache_clear()
            _build_radar_name_to_code_map.cache_clear()


class TestMissingLevels:
    """Levels absent from a VVP file are filled onto a contiguous grid."""

    MISSING_ASL = (500, 1300, 1700)

    def test_absent_levels_are_filled_as_no_echo(self):
        _, df = _parse_vvp_file(GAPPED_VVP_FILE)
        ds = read_vvp(GAPPED_VVP_FILE)
        antenna = ds.attrs["antenna_height_m"]
        missing = [height - antenna for height in self.MISSING_ASL]

        np.testing.assert_array_equal(np.diff(ds.height), 200)
        assert int(ds.height[0]) == 300 - antenna
        assert int(ds.height[-1]) == 5100 - antenna
        assert ds.attrs["missing_levels_m"] == missing

        gap = ds.sel(height=missing)
        np.testing.assert_array_equal(gap.zcount, 0)
        np.testing.assert_array_equal(gap.sample_count, 0)
        for variable in ("lin_dbz", "log_dbz", "corrected_dbz"):
            np.testing.assert_array_equal(gap[variable], MDS)
        assert gap.wind_speed.isnull().all()
        assert np.issubdtype(ds.zcount.dtype, np.integer)

        observed = df[df.height > antenna]
        measured = ds.sel(height=observed.height.values - antenna)
        np.testing.assert_array_equal(measured.lin_dbz, observed.lin_dbz)
        np.testing.assert_array_equal(measured.zcount, observed.zcount)

    def test_complete_profile_has_no_missing_levels(self):
        ds = read_vvp(SAMPLE_VVP_FILE)
        assert ds.attrs["missing_levels_m"] == []

    def test_heights_off_the_profile_grid_raise(self, tmp_path):
        path = tmp_path / "off_grid.txt"
        row = " 30 1000 0 0 0 0 0 0 10 0 0 0 0 0 0 0 0 0"
        path.write_text(
            "KANKAANPAA VVP_40 PPI1_A 2025 08 24 11 00 ELEVS: 0.7\n\n"
            "Height Count ZCnt ...\n"
            + "\n".join(f"{h}{row}" for h in (300, 600, 700))
            + "\n"
        )
        with pytest.raises(ValueError, match="regular 200 m grid"):
            read_vvp(path)
