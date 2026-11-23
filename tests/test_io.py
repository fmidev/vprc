#!/usr/bin/env python3
"""
Tests for VVP file parsing functionality.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vprc.config import list_radar_codes, resolve_radar_code
from vprc.constants import MDS
from vprc.exceptions import NoUsableProfileError
from vprc.io import (
    VVPHeader,
    _build_radar_name_to_code_map,
    _get_radar_metadata,
    _load_radar_defaults,
    _parse_vvp_file,
    _parse_vvp_header,
    _vvp_dataframe_to_xarray,
    height_to_sea_level,
    read_vvp,
)

# Test data path
TEST_DATA_DIR = Path(__file__).parent / "data"
SAMPLE_VVP_FILE = TEST_DATA_DIR / "202508241100_KAN.VVP_40.txt"
KOR_VVP_FILE = TEST_DATA_DIR / "202308281000_KOR.VVP_40.txt"
# Kankaanpää 2025-08-24 11 UTC without its lowest (300 m ASL) level.
NO_LOWEST_VVP_FILE = TEST_DATA_DIR / "variant_KAN_no_lowest_level.VVP_40.txt"
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
        header, df = _parse_vvp_file(SAMPLE_VVP_FILE)

        # Check header
        assert header.radar == "KANKAANPAA"
        assert header.product == "VVP_40"
        assert header.timestamp == datetime(2025, 8, 24, 11, 0)
        assert header.elevation_angles == [0.7, 1.5, 3.0]

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 35  # Number of height levels in sample file
        assert df["height"].min() == 100
        assert df["height"].max() == 6900

    def test_dataframe_columns(self):
        """Test that DataFrame has expected columns."""
        _, df = _parse_vvp_file(SAMPLE_VVP_FILE)

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

        assert list(df.columns) == expected_cols

    def test_dataframe_sorted_by_height(self):
        """Test that DataFrame is sorted by height ascending."""
        _, df = _parse_vvp_file(SAMPLE_VVP_FILE)

        # Check that heights are monotonically increasing
        assert df["height"].is_monotonic_increasing

    def test_dataframe_height_dtype(self):
        """Test that height column is integer type."""
        _, df = _parse_vvp_file(SAMPLE_VVP_FILE)

        assert df["height"].dtype == int

    def test_dataframe_values(self):
        """Test specific data values from sample file."""
        _, df = _parse_vvp_file(SAMPLE_VVP_FILE)

        # Check first row (height=100 in sorted data)
        row_100 = df[df["height"] == 100].iloc[0]
        assert row_100["count"] == 1326
        assert row_100["zcount"] == 1326
        assert abs(row_100["wind_speed"] - 5.5) < 0.01
        assert abs(row_100["direction"] - 350.0) < 0.01

        # Check a middle row (height=1500)
        row_1500 = df[df["height"] == 1500].iloc[0]
        assert row_1500["count"] == 5000
        assert abs(row_1500["wind_speed"] - 7.5) < 0.01
        assert abs(row_1500["lin_dbz"] - 30.0) < 0.1

    def test_parse_nonexistent_file(self):
        """Test that parsing nonexistent file raises FileNotFoundError."""
        nonexistent = TEST_DATA_DIR / "nonexistent_file.txt"
        with pytest.raises(FileNotFoundError):
            _parse_vvp_file(nonexistent)

    @pytest.mark.parametrize(
        "body",
        ["", "\n", "\nHeight Count ZCnt ...\n", "\nHeight Count ZCnt ...\n\n  \n"],
    )
    def test_header_only_file_has_no_usable_profile(self, tmp_path: Path, body):
        path = tmp_path / "header_only.txt"
        path.write_text(
            "KANKAANPAA VVP_40 PPI1_A 2025 08 24 11 00 ELEVS:  0.7 1.5 3.0\n" + body
        )
        with pytest.raises(NoUsableProfileError, match="no profile levels"):
            _parse_vvp_file(path)

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ("", "Empty VVP file"),
            ("RADAR VVP_40 PPI1_A 2025 08 24 11 00\n\n", "Invalid header"),
        ],
    )
    def test_empty_or_headerless_file_is_invalid(
        self, tmp_path: Path, content, message
    ):
        path = tmp_path / "invalid.txt"
        path.write_text(content)
        with pytest.raises(ValueError, match=message) as error:
            _parse_vvp_file(path)
        assert not isinstance(error.value, NoUsableProfileError)


class TestVVPDataframeToXarray:
    """Tests for vvp_dataframe_to_xarray function."""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample parsed data."""
        return _parse_vvp_file(SAMPLE_VVP_FILE)

    def test_heights_above_antenna(self, sample_data: tuple[VVPHeader, pd.DataFrame]):
        """KAN (antenna 174 m): 300-6900 m ASL become 126-6726 m above antenna.

        The 100 m ASL level is below the antenna and is dropped.
        """
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        np.testing.assert_array_equal(ds.height, np.arange(126, 6727, 200))

    def test_xarray_variables(self, sample_data: tuple[VVPHeader, pd.DataFrame]):
        """Test that xarray has expected data variables."""
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        expected_vars = [
            "sample_count",
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
            "corrected_dbz",  # Added by conversion function
        ]

        for var in expected_vars:
            assert var in ds.data_vars, f"Variable {var} not in dataset"

    def test_xarray_corrected_dbz(self, sample_data: tuple[VVPHeader, pd.DataFrame]):
        """Test that corrected_dbz is initialized as copy of lin_dbz."""
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        np.testing.assert_array_equal(ds["corrected_dbz"], ds["lin_dbz"])
        # Processing updates corrected_dbz; the raw profile must not follow.
        assert not np.shares_memory(ds["corrected_dbz"].values, ds["lin_dbz"].values)

    def test_xarray_metadata(self, sample_data: tuple[VVPHeader, pd.DataFrame]):
        """Test that xarray has correct metadata attributes."""
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header, source_file=SAMPLE_VVP_FILE)

        assert ds.attrs["radar"] == "KANKAANPAA"
        assert ds.attrs["product"] == "VVP_40"
        assert ds.attrs["scan_type"] == "PPI1_A"
        assert ds.attrs["timestamp"] == "2025-08-24T11:00:00"
        assert ds.attrs["elevation_angles"] == [0.7, 1.5, 3.0]
        assert ds.attrs["antenna_height_m"] == 174
        assert ds.attrs["lowest_level_offset_m"] == 126
        assert ds.attrs["source_file"] == str(SAMPLE_VVP_FILE)

    def test_xarray_data_access(self, sample_data: tuple[VVPHeader, pd.DataFrame]):
        """Test accessing data from xarray Dataset.

        Heights are above antenna, so 1500m ASL -> 1326m above antenna.
        """
        header, df = sample_data
        ds = _vvp_dataframe_to_xarray(df, header)

        # Test selection by height (above antenna)
        # Original 1500m ASL - 174m antenna = 1326m above antenna
        data_at_1326 = ds.sel(height=1326)
        assert data_at_1326["sample_count"].values == 5000
        assert abs(float(data_at_1326["wind_speed"].values) - 7.5) < 0.01


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_no_levels_above_antenna_has_no_usable_profile(self):
        header, df = _parse_vvp_file(SAMPLE_VVP_FILE)
        with pytest.raises(NoUsableProfileError, match="no levels above"):
            _vvp_dataframe_to_xarray(df[df.height < 174], header)

    def test_no_levels_from_lowest_level_has_no_usable_profile(self):
        """Levels below the lowest profile level are unused, as in Perl."""
        header, df = _parse_vvp_file(SAMPLE_VVP_FILE)
        geometry = {"antenna_height_m": 174, "lowest_level_offset_m": 326}
        with pytest.raises(NoUsableProfileError, match="at or above"):
            _vvp_dataframe_to_xarray(df[df.height <= 300], header, geometry)

    @pytest.mark.parametrize(
        "freezing_level", [float("nan"), float("inf"), "2000", True]
    )
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

    def test_missing_freezing_level(self):
        """Test handling when freezing_level_m is not provided or invalid.

        Three cases:
        1. Unknown (None/not provided): Keep as None
        2. No freezing layer (0 or negative): Normalize to 0
        3. Valid (positive ASL): Convert to above-antenna
        """
        # KAN antenna height is 174m
        antenna_height = 174

        # Case 1: Unknown - No metadata provided
        ds1 = read_vvp(SAMPLE_VVP_FILE)
        assert ds1.attrs.get("freezing_level_m") is None

        # Case 1: Unknown - freezing_level_m is None
        ds2 = read_vvp(SAMPLE_VVP_FILE, radar_metadata={"freezing_level_m": None})
        assert ds2.attrs.get("freezing_level_m") is None

        # Case 2: No freezing layer - negative (below surface)
        ds3 = read_vvp(SAMPLE_VVP_FILE, radar_metadata={"freezing_level_m": -100})
        assert ds3.attrs.get("freezing_level_m") == 0

        # Case 2: No freezing layer - zero (at sea level)
        ds4 = read_vvp(SAMPLE_VVP_FILE, radar_metadata={"freezing_level_m": 0})
        assert ds4.attrs.get("freezing_level_m") == 0

        # Case 2b: Freezing level positive ASL but below antenna (e.g., 100m ASL < 174m antenna)
        # No melting layer visible to radar → normalized to 0
        ds4b = read_vvp(SAMPLE_VVP_FILE, radar_metadata={"freezing_level_m": 100})
        assert ds4b.attrs.get("freezing_level_m") == 0

        # Case 3: Valid freezing layer - positive ASL value
        # Input 2000m ASL -> stored as 2000 - 174 = 1826m above antenna
        ds5 = read_vvp(SAMPLE_VVP_FILE, radar_metadata={"freezing_level_m": 2000})
        assert ds5.attrs["freezing_level_m"] == 2000 - antenna_height


class TestRadarConfiguration:
    """Tests for radar configuration loading and precedence."""

    def test_load_radar_defaults(self):
        """Test loading the package default radar configuration."""
        config = _load_radar_defaults()

        # Check that all expected radar codes are present
        assert "KAN" in config
        assert "VAN" in config
        assert "defaults" in config

        # Check structure of a known radar
        assert "antenna_height_m" in config["KAN"]
        assert "lowest_level_offset_m" in config["KAN"]

        # Check structure of defaults section; geometry has no default
        assert "antenna_height_m" not in config["defaults"]
        assert "lowest_level_offset_m" not in config["defaults"]
        assert "beamwidth_deg" in config["defaults"]

        # Verify known values from Perl reference
        assert config["KAN"]["antenna_height_m"] == 174
        assert config["KAN"]["lowest_level_offset_m"] == 126

    def test_get_radar_metadata_known_radar(self):
        """Test retrieving metadata for a known radar code."""
        meta = _get_radar_metadata("KAN")

        assert meta["antenna_height_m"] == 174
        assert meta["lowest_level_offset_m"] == 126

    def test_get_radar_metadata_unknown_radar_has_no_geometry(self):
        """[defaults] does not provide geometry for unknown radar codes."""
        with pytest.raises(
            ValueError, match="UNKNOWN does not define antenna_height_m"
        ):
            _get_radar_metadata("UNKNOWN")

    def test_unknown_radar_with_full_geometry_override(self):
        meta = _get_radar_metadata(
            "UNKNOWN", {"antenna_height_m": 100, "lowest_level_offset_m": 200}
        )
        assert (meta["antenna_height_m"], meta["lowest_level_offset_m"]) == (100, 200)
        defaults = _load_radar_defaults()["defaults"]
        assert meta["beamwidth_deg"] == defaults["beamwidth_deg"]

    def test_get_radar_metadata_with_override(self):
        """Test that override metadata takes precedence."""
        meta = _get_radar_metadata(
            "KAN", {"antenna_height_m": 999, "lowest_level_offset_m": 1}
        )
        assert meta["antenna_height_m"] == 999
        assert meta["lowest_level_offset_m"] == 1

    @pytest.mark.parametrize("key", ["antenna_height_m", "lowest_level_offset_m"])
    def test_geometry_must_be_overridden_together(self, key):
        with pytest.raises(ValueError, match=f"overridden together; only {key}"):
            _get_radar_metadata("KAN", {key: 200})

    @pytest.mark.parametrize("missing", ["antenna_height_m", "lowest_level_offset_m"])
    def test_radar_section_must_define_geometry(
        self, tmp_path, monkeypatch, caplog, missing
    ):
        """Radar sections never take geometry from [defaults]."""
        geometry = {"antenna_height_m": 174, "lowest_level_offset_m": 126}
        del geometry[missing]
        config = tmp_path / "partial.toml"
        config.write_text(
            "[defaults]\nantenna_height_m = 198\nlowest_level_offset_m = 102\n"
            '[KAN]\nname = "KANKAANPAA"\n'
            + "".join(f"{key} = {value}\n" for key, value in geometry.items())
        )
        monkeypatch.setenv("VPRC_RADAR_CONFIG", str(config))
        _load_radar_defaults.cache_clear()
        _build_radar_name_to_code_map.cache_clear()
        try:
            with pytest.raises(ValueError, match=f"KAN does not define {missing}"):
                read_vvp(SAMPLE_VVP_FILE)
            assert (
                "Ignoring antenna_height_m, lowest_level_offset_m in [defaults]"
                in caplog.text
            )
        finally:
            monkeypatch.delenv("VPRC_RADAR_CONFIG")
            _load_radar_defaults.cache_clear()
            _build_radar_name_to_code_map.cache_clear()

    @pytest.mark.parametrize("key", ["antenna_height_m", "lowest_level_offset_m"])
    @pytest.mark.parametrize("value", [174.5, float("nan"), "174", True])
    def test_geometry_must_be_whole_meters(self, key, value):
        geometry = {"antenna_height_m": 174, "lowest_level_offset_m": 126, key: value}
        with pytest.raises(ValueError, match=f"{key} must be whole meters"):
            read_vvp(SAMPLE_VVP_FILE, geometry)

    @pytest.mark.parametrize("offset", [0, -26])
    def test_lowest_level_must_be_above_antenna(self, offset):
        with pytest.raises(ValueError, match="must be above the antenna"):
            read_vvp(
                SAMPLE_VVP_FILE,
                {"antenna_height_m": 174, "lowest_level_offset_m": offset},
            )

    def test_missing_config_file_raises(self, tmp_path, monkeypatch):
        """A misconfigured container must not fall back to packaged defaults."""
        monkeypatch.setenv("VPRC_RADAR_CONFIG", str(tmp_path / "missing.toml"))
        _load_radar_defaults.cache_clear()
        try:
            with pytest.raises(FileNotFoundError, match="VPRC_RADAR_CONFIG"):
                _load_radar_defaults()
        finally:
            monkeypatch.delenv("VPRC_RADAR_CONFIG")
            _load_radar_defaults.cache_clear()

    def test_get_radar_metadata_with_additional_fields(self):
        """Test adding additional fields via override."""
        meta = _get_radar_metadata("KAN", {"freezing_level_m": 2000})

        # Original fields from TOML
        assert meta["antenna_height_m"] == 174
        assert meta["lowest_level_offset_m"] == 126
        # New field from override
        assert meta["freezing_level_m"] == 2000

    def test_get_radar_metadata_includes_beamwidth(self):
        """Test that beamwidth_deg is loaded from [defaults] section."""
        meta = _get_radar_metadata("KAN")

        # Should include beamwidth from [defaults] section
        assert "beamwidth_deg" in meta
        defaults = _load_radar_defaults()["defaults"]
        assert meta["beamwidth_deg"] == defaults["beamwidth_deg"]

    def test_get_radar_metadata_beamwidth_can_be_overridden(self):
        """Test that beamwidth_deg can be overridden per-radar."""
        meta = _get_radar_metadata("KAN", {"beamwidth_deg": 1.0})

        assert meta["beamwidth_deg"] == 1.0

    def test_list_radar_codes_returns_enabled_by_default(self):
        """Test that list_radar_codes returns enabled radars by default."""
        codes = list_radar_codes()

        # All radars in default config are enabled
        assert "KAN" in codes
        assert "VAN" in codes
        assert "VIH" in codes
        # Should not include special sections
        assert "defaults" not in codes
        assert "network" not in codes

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

        old_env = os.environ.get("VPRC_RADAR_CONFIG")
        try:
            os.environ["VPRC_RADAR_CONFIG"] = str(custom_config)
            _load_radar_defaults.cache_clear()
            _build_radar_name_to_code_map.cache_clear()

            # enabled_only=True (default) should exclude VAN
            codes_enabled = list_radar_codes(enabled_only=True)
            assert "KAN" in codes_enabled
            assert "VIH" in codes_enabled
            assert "VAN" not in codes_enabled

            # enabled_only=False should include all
            codes_all = list_radar_codes(enabled_only=False)
            assert "KAN" in codes_all
            assert "VIH" in codes_all
            assert "VAN" in codes_all

        finally:
            if old_env is not None:
                os.environ["VPRC_RADAR_CONFIG"] = old_env
            else:
                os.environ.pop("VPRC_RADAR_CONFIG", None)
            _load_radar_defaults.cache_clear()
            _build_radar_name_to_code_map.cache_clear()

    def test_read_vvp_with_defaults(self):
        """Test that read_vvp loads TOML defaults automatically."""
        ds = read_vvp(SAMPLE_VVP_FILE)

        # Should have loaded KAN radar defaults
        assert ds.attrs["radar_code"] == "KAN"
        assert ds.attrs["antenna_height_m"] == 174
        assert ds.attrs["lowest_level_offset_m"] == 126
        # Should also include beamwidth from [defaults] section
        assert ds.attrs["beamwidth_deg"] == 0.95

    def test_read_vvp_with_override(self):
        """Test precedence: function parameter > TOML defaults."""
        # Override the geometry (as a pair) and the freezing level (m ASL)
        ds = read_vvp(
            SAMPLE_VVP_FILE,
            {
                "antenna_height_m": 200,
                "lowest_level_offset_m": 100,
                "freezing_level_m": 2500,
            },
        )

        # Overridden values
        assert ds.attrs["antenna_height_m"] == 200
        assert ds.attrs["lowest_level_offset_m"] == 100
        assert int(ds.height[0]) == 100
        # TOML value for a field not overridden
        assert ds.attrs["latitude"] == pytest.approx(61.81085)
        # freezing_level converted from ASL (2500) to above-antenna (2500 - 200 = 2300)
        assert ds.attrs["freezing_level_m"] == 2300

    def test_precedence_order(self, tmp_path):
        """Runtime metadata overrides the selected TOML.

        VPRC_RADAR_CONFIG replaces the packaged configuration rather than
        merging with it.
        """
        # Create custom config
        custom_config = tmp_path / "test_radars.toml"
        custom_config.write_text("""
[KAN]
name = "KANKAANPAA"
antenna_height_m = 500
lowest_level_offset_m = 400
beamwidth_deg = 1.2
custom_field = 123
""")

        old_env = os.environ.get("VPRC_RADAR_CONFIG")
        try:
            os.environ["VPRC_RADAR_CONFIG"] = str(custom_config)
            _load_radar_defaults.cache_clear()
            _build_radar_name_to_code_map.cache_clear()

            # Load with function override
            ds = read_vvp(
                SAMPLE_VVP_FILE,
                {
                    "antenna_height_m": 300,
                    "lowest_level_offset_m": 200,
                    "custom_field": 7,
                },
            )

            # Function parameters win
            assert ds.attrs["antenna_height_m"] == 300
            assert ds.attrs["lowest_level_offset_m"] == 200
            assert ds.attrs["custom_field"] == 7
            # Custom TOML provides this
            assert ds.attrs["beamwidth_deg"] == 1.2
            # Nothing comes from the packaged file
            assert "latitude" not in ds.attrs
            assert "max_range_km" not in ds.attrs

        finally:
            if old_env is not None:
                os.environ["VPRC_RADAR_CONFIG"] = old_env
            else:
                os.environ.pop("VPRC_RADAR_CONFIG", None)
            _load_radar_defaults.cache_clear()
            _build_radar_name_to_code_map.cache_clear()


class TestRadarNameMapping:
    """Tests for radar name to code mapping from TOML configuration."""

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

    @pytest.mark.parametrize(
        ("header_name", "code"),
        [
            # Perl reference names of radars not in the list above
            ("VANTAA", "VAN"),
            ("KERAVA", "KER"),
            ("IKAALINEN", "IKA"),
            # Korppoo's older site name
            ("KOR", "KOR"),
            ("kankaanpaa", "KAN"),
        ],
    )
    def test_legacy_and_lowercase_header_names_resolve(self, header_name, code):
        assert resolve_radar_code(header_name) == code

    def test_aliases_must_be_a_list(self, tmp_path, monkeypatch):
        config = tmp_path / "aliases.toml"
        config.write_text('[KOR]\nname = "KORPPOO"\naliases = "KOR"\n')
        monkeypatch.setenv("VPRC_RADAR_CONFIG", str(config))
        _load_radar_defaults.cache_clear()
        _build_radar_name_to_code_map.cache_clear()
        try:
            with pytest.raises(TypeError, match="'aliases' must be a list"):
                _build_radar_name_to_code_map()
        finally:
            monkeypatch.delenv("VPRC_RADAR_CONFIG")
            _load_radar_defaults.cache_clear()
            _build_radar_name_to_code_map.cache_clear()

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
        with pytest.raises(ValueError, match="not on the 200 m grid"):
            read_vvp(path)

    def test_missing_lowest_level_keeps_configured_lowest_level(self):
        """Regression: the first level present in the file became 'ground'."""
        ds = read_vvp(NO_LOWEST_VVP_FILE)
        assert ds.attrs["lowest_level_offset_m"] == 126
        assert int(ds.height[0]) == 126
        assert ds.attrs["missing_levels_m"] == [126]
        assert int(ds.zcount.sel(height=126)) == 0
        assert float(ds.lin_dbz.sel(height=126)) == MDS

    def test_geometry_inconsistent_with_file_raises(self):
        with pytest.raises(ValueError, match="check antenna_height_m and lowest_level"):
            read_vvp(
                SAMPLE_VVP_FILE,
                {"antenna_height_m": 174, "lowest_level_offset_m": 100},
            )


class TestHeightToSeaLevel:
    def test_converts_back_to_asl(self):
        ds = read_vvp(SAMPLE_VVP_FILE)
        np.testing.assert_array_equal(
            height_to_sea_level(ds), np.arange(300, 6901, 200)
        )

    def test_missing_antenna_height_raises(self):
        ds = read_vvp(SAMPLE_VVP_FILE)
        del ds.attrs["antenna_height_m"]
        with pytest.raises(KeyError, match="antenna_height_m"):
            height_to_sea_level(ds)
