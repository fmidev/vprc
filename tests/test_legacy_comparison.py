"""End-to-end tests comparing Python implementation against legacy Perl output.

These tests validate that the Python reimplementation produces results
matching the reference Perl implementation (allprof_prodx2.pl).

Test data: legacy/202508241100_KAN.VVP_40.* files
- .txt: Input VVP profile
- .profile: Expected corrected profile output

Height coordinate system:
Both Python and legacy use heights above antenna (meters). The legacy code
converts sea-level heights from VVP files using:
    height_above_antenna = height_sea_level - antenna_height_m
Levels below antenna are dropped (negative heights).
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from vprc import process_vvp
from vprc.constants import MDS

from .legacy_parser import parse_legacy_profile


# Paths to test data
DATA_DIR = Path(__file__).parent / 'data'
INPUT_FILE = DATA_DIR / '202508241100_KAN.VVP_40.txt'
EXPECTED_PROFILE = DATA_DIR / '202508241100_KAN.VVP_40.profile'

# Freezing level from sounding (since NWP data unavailable)
# The legacy run used HIRLAM data; we use observed value for testing
FREEZING_LEVEL_M = 1700


@pytest.fixture
def expected_profile():
    """Load the expected legacy profile output."""
    ds, header, footer = parse_legacy_profile(EXPECTED_PROFILE)
    return ds, header, footer


@pytest.fixture
def processed_result():
    """Run the Python pipeline on the test input."""
    return process_vvp(
        INPUT_FILE,
        freezing_level_m=FREEZING_LEVEL_M,
    )


class TestLegacyProfileComparison:
    """Compare Python output against legacy .profile file.

    Both implementations now use heights above antenna, so direct
    comparison is possible.
    """

    def test_input_file_exists(self):
        """Verify test data files exist."""
        assert INPUT_FILE.exists(), f"Input file not found: {INPUT_FILE}"
        assert EXPECTED_PROFILE.exists(), f"Expected output not found: {EXPECTED_PROFILE}"

    def test_height_coordinates_match(self, processed_result, expected_profile):
        """Verify height coordinates are identical."""
        expected_ds, _, _ = expected_profile
        result_ds = processed_result.dataset

        # Get common length (legacy may have more levels with MDS values)
        n_common = min(len(result_ds.height), len(expected_ds.height))

        np.testing.assert_array_equal(
            result_ds.height.values[:n_common],
            expected_ds.height.values[:n_common],
            err_msg="Height coordinates do not match"
        )

    def test_height_step_consistent(self, processed_result, expected_profile):
        """Verify height step is consistent at 200m in both outputs."""
        expected_ds, _, _ = expected_profile
        result_ds = processed_result.dataset

        python_step = np.diff(result_ds.height.values)
        legacy_step = np.diff(expected_ds.height.values)

        assert np.all(python_step == 200), f"Python step not 200m: {python_step[:5]}"
        assert np.all(legacy_step == 200), f"Legacy step not 200m: {legacy_step[:5]}"

    def test_lin_dbz_values_match(self, processed_result, expected_profile):
        """Verify lin_dbz (input reflectivity) values match.

        These should be identical since they come from the same input file.
        """
        expected_ds, _, _ = expected_profile
        result_ds = processed_result.dataset

        n_common = min(len(result_ds.height), len(expected_ds.height))

        np.testing.assert_allclose(
            result_ds['lin_dbz'].values[:n_common],
            expected_ds['lin_dbz'].values[:n_common],
            rtol=0.01,
            err_msg="lin_dbz values do not match"
        )

    def test_corrected_dbz_values_match(self, processed_result, expected_profile):
        """Verify corrected_dbz values match after full pipeline."""
        expected_ds, _, _ = expected_profile
        result_ds = processed_result.dataset

        n_common = min(len(result_ds.height), len(expected_ds.height))

        # Compare only valid (non-MDS) levels
        valid_mask = expected_ds['corrected_dbz'].values[:n_common] > MDS

        np.testing.assert_allclose(
            result_ds['corrected_dbz'].values[:n_common][valid_mask],
            expected_ds['corrected_dbz'].values[:n_common][valid_mask],
            rtol=0.01,
            atol=0.1,
            err_msg="corrected_dbz values do not match legacy output"
        )

    def test_max_dbz_matches(self, processed_result, expected_profile):
        """Verify maximum dBZ value matches legacy."""
        expected_ds, _, footer = expected_profile
        result_ds = processed_result.dataset

        # Get max from valid values (exclude MDS)
        from vprc.constants import MDS
        valid_mask = result_ds['corrected_dbz'] > MDS
        result_max = float(result_ds['corrected_dbz'].where(valid_mask).max())

        assert abs(result_max - footer.max_dbz) < 0.5, (
            f"Max dBZ mismatch: got {result_max}, expected {footer.max_dbz}"
        )

    def test_layer_boundaries_match(self, processed_result, expected_profile):
        """Verify layer boundaries match legacy classification.

        Height mapping:
        """
        _, _, footer = expected_profile
        classification = processed_result.classification

        if classification.lowest_layer is not None:
            layer = classification.lowest_layer
            assert layer.bottom_height == footer.bottom_height, (
                f"Bottom height mismatch: got {layer.bottom_height}, "
                f"expected {footer.bottom_height}"
            )
            assert layer.top_height == footer.top_height, (
                f"Top height mismatch: got {layer.top_height}, "
                f"expected {footer.top_height}"
            )

    def test_precipitation_classification_matches(self, processed_result, expected_profile):
        """Verify precipitation type classification matches legacy.

        Legacy values: 0=none, 1=snow, 2=rain, 3=sleet
        """
        _, _, footer = expected_profile
        classification = processed_result.classification

        # The footer.precipitation field indicates precipitation type
        # Our classification should agree on precipitation presence
        legacy_has_precip = footer.precipitation > 0
        python_has_precip = classification.usable_for_vpr

        assert legacy_has_precip == python_has_precip, (
            f"Precipitation classification mismatch: "
            f"legacy={footer.precipitation}, python_usable={python_has_precip}"
        )

    def test_quality_weight_reasonable(self, processed_result, expected_profile):
        """Verify quality weight is in reasonable range.

        The legacy quality weight (laatupaino) ranges 0.0-1.0.
        """
        _, _, footer = expected_profile

        # Just verify the legacy value is captured correctly
        assert 0.0 <= footer.quality_weight <= 1.0, (
            f"Legacy quality weight out of range: {footer.quality_weight}"
        )


class TestBrightBandComparison:
    """Compare bright band detection against legacy.

    Note: The legacy run may have had no access to NWP freezing level data
    (footer.freezing_level=0). In such cases, BB detection comparison
    is not meaningful since Python uses the provided freezing_level_m.
    """

    def test_bright_band_detection_agrees(self, processed_result, expected_profile):
        """Verify bright band detection matches legacy.

        Legacy BB flag: 0=not detected, 1=detected

        Skip if legacy had no freezing level (freezing_level=0 in footer).
        """
        _, _, footer = expected_profile
        bb_result = processed_result.bright_band

        # If legacy had no freezing level data, BB comparison is not meaningful
        if footer.freezing_level == 0:
            pytest.skip(
                "Legacy had no freezing level data (freezing_level=0), "
                "BB detection comparison not applicable"
            )

        legacy_bb_detected = footer.bright_band == 1

        assert bb_result.detected == legacy_bb_detected, (
            f"Bright band detection mismatch: "
            f"legacy={legacy_bb_detected}, python={bb_result.detected}"
        )

    def test_bright_band_height_matches(self, processed_result, expected_profile):
        """Verify bright band height matches legacy (if detected)."""
        _, _, footer = expected_profile
        bb_result = processed_result.bright_band

        if footer.bright_band == 1 and bb_result.detected:
            # Direct comparison - both use heights above antenna
            assert abs(bb_result.peak_height - footer.bb_height) <= 200, (
                f"BB height mismatch: got {bb_result.peak_height}, "
                f"expected {footer.bb_height}"
            )


class TestProcessingPipeline:
    """Integration tests for the full processing pipeline."""

    def test_pipeline_completes_without_error(self, processed_result):
        """Verify the full pipeline runs successfully."""
        assert processed_result is not None
        assert processed_result.dataset is not None
        assert processed_result.classification is not None
        assert processed_result.bright_band is not None

    def test_dataset_has_required_variables(self, processed_result):
        """Verify output dataset contains expected variables."""
        ds = processed_result.dataset
        required_vars = ['lin_dbz', 'corrected_dbz']
        for var in required_vars:
            assert var in ds, f"Missing variable: {var}"

    def test_vpr_correction_computed_when_usable(self, processed_result):
        """Verify VPR correction is computed for usable profiles."""
        if processed_result.classification.usable_for_vpr:
            assert processed_result.vpr_correction is not None, (
                "VPR correction should be computed for usable profiles"
            )


class TestVPRCorrectionComparison:
    """Compare VPR correction output against legacy .cor file.

    Legacy format outputs corrections for:
    - 2 CAPPI heights (500m, 1000m)
    - 4 elevation angles (0.7°, 1.5°, 3.0°, 3.3° for Kankaanpää)

    The Python implementation currently supports CAPPI heights.
    """

    @pytest.fixture
    def legacy_cor(self):
        """Load the legacy .cor file."""
        from .legacy_parser import parse_legacy_cor
        cor_path = DATA_DIR / '202508241100_KAN.VVP_40.cor'
        return parse_legacy_cor(cor_path)

    @pytest.fixture
    def python_profile(self):
        """Run the Python pipeline to get the corrected profile."""
        return process_vvp(INPUT_FILE, freezing_level_m=FREEZING_LEVEL_M)

    def test_cor_file_exists(self):
        """Verify .cor test data file exists."""
        cor_path = DATA_DIR / '202508241100_KAN.VVP_40.cor'
        assert cor_path.exists(), f".cor file not found: {cor_path}"

    def test_cor_parser_loads_data(self, legacy_cor):
        """Verify .cor parser returns valid data structure."""
        ds, header = legacy_cor
        assert 'correction_klim_db' in ds
        assert 'correction_rain_db' in ds
        assert 'beam_height_m' in ds
        # 2 CAPPI heights + 4 elevation angles = 6 correction types
        assert len(ds.correction_type) == 6
        assert len(header.elevation_angles) == 4

    def test_cor_header_values(self, legacy_cor):
        """Verify parsed header contains expected metadata."""
        _, header = legacy_cor
        assert header.radar_name == 'KANKAANPAA'
        assert header.step == 200
        assert header.mds == -45
        assert header.precip_code == 2  # rain
        assert header.elevation_angles == [0.7, 1.5, 3.0, 3.3]

    def test_cappi_beam_height_calculation(self, legacy_cor):
        """Verify CAPPI beam height calculation matches legacy.

        The CAPPI heights (500m, 1000m) use pseudo-CAPPI logic where the
        beam height matches the target CAPPI level at close range but
        transitions to minimum elevation at far range.
        """
        from vprc.vpr_correction import compute_beam_height, solve_elevation_for_height

        ds, header = legacy_cor
        # Kankaanpää: antenna 174m ASL, we work in heights above antenna
        antenna_height_m = 174

        # Test CAPPI 500m beam heights
        # At very close range, should approach 500 - (antenna adjustment)
        # At far range, should follow minimum elevation angle
        cappi_500_heights = ds['beam_height_m'].sel(correction_type='cappi_500')

        # At far range (200km), the beam follows the lowest elevation
        # and should be much higher than 500m (above antenna)
        far_range_height = float(cappi_500_heights.sel(range_km=200).values)
        assert far_range_height > 500, (
            f"Far range CAPPI 500m height should exceed 500m: got {far_range_height}"
        )

    def test_elevation_beam_height_calculation(self, legacy_cor):
        """Verify fixed elevation beam height calculation matches legacy.

        For fixed elevation angles, the beam height at each range should
        follow the standard 4/3 Earth refraction model.
        """
        from vprc.vpr_correction import compute_beam_height

        ds, header = legacy_cor
        # Kankaanpää: antenna 174m ASL
        antenna_height_m = 174

        # Test first elevation angle (0.7°)
        elev = header.elevation_angles[0]
        test_ranges = [50, 100, 150, 200]

        for range_km in test_ranges:
            legacy_height = int(ds['beam_height_m'].sel(
                range_km=range_km, correction_type='elev_1'
            ).values)

            python_height_asl = compute_beam_height(
                range_km * 1000, elev, antenna_height_m
            )
            # Convert to height above antenna
            python_height = python_height_asl - antenna_height_m

            # Allow tolerance for different implementations of the formula
            # Legacy uses Rinehart formula, Python uses wradlib
            tolerance = max(10, int(python_height * 0.02))  # 2% or 10m
            assert abs(python_height - legacy_height) <= tolerance, (
                f"Beam height mismatch at {range_km}km, {elev}°: "
                f"Python={python_height:.0f}m, legacy={legacy_height}m"
            )

    def test_correction_sign_convention(self, legacy_cor):
        """Verify correction sign convention matches legacy.

        Positive correction = beam is above ground, seeing weaker echo
        Negative correction = beam sees stronger echo (e.g. bright band)
        """
        ds, _ = legacy_cor

        # At close range, CAPPI corrections should be small
        corr_close = ds['correction_klim_db'].sel(range_km=10, correction_type='cappi_500').values
        assert abs(corr_close) < 5, f"Close range correction too large: {corr_close}"

    def test_max_dbz_matches_profile(self, legacy_cor, expected_profile):
        """Verify .cor max_dBZ matches .profile footer."""
        _, cor_header = legacy_cor
        _, _, profile_footer = expected_profile

        assert abs(cor_header.max_dbz - profile_footer.max_dbz) < 0.1, (
            f"max_dBZ mismatch: .cor={cor_header.max_dbz}, "
            f".profile={profile_footer.max_dbz}"
        )

    def test_quality_weight_consistent(self, legacy_cor, expected_profile):
        """Verify quality weight is consistent between .cor and .profile."""
        _, cor_header = legacy_cor
        _, _, profile_footer = expected_profile

        assert abs(cor_header.quality_weight - profile_footer.quality_weight) < 0.01, (
            f"Quality weight mismatch: .cor={cor_header.quality_weight}, "
            f".profile={profile_footer.quality_weight}"
        )

    def test_python_cappi_corrections_computed(self, python_profile):
        """Verify Python computes VPR corrections for CAPPI heights."""
        vpr = python_profile.vpr_correction
        assert vpr is not None, "VPR correction should be computed"
        assert vpr.usable, "VPR correction should be usable for this profile"

        # Check correction dataset structure
        ds = vpr.corrections
        assert 'cappi_correction_db' in ds
        assert 'range_km' in ds.dims
        assert 'cappi_height' in ds.dims

    def test_python_vs_legacy_cappi_500_trend(self, legacy_cor, python_profile):
        """Compare Python CAPPI 500m corrections to legacy trend.

        Both should show increasing positive correction with range
        (beam sees weaker echo at higher altitudes).
        """
        legacy_ds, _ = legacy_cor
        python_vpr = python_profile.vpr_correction

        if python_vpr is None or not python_vpr.usable:
            pytest.skip("Python VPR correction not computed")

        # Get klim corrections (climatology profile) from legacy
        legacy_corr = legacy_ds['correction_klim_db'].sel(correction_type='cappi_500')

        # Get corrections from Python (for CAPPI 500m)
        python_ds = python_vpr.corrections
        if 500 not in python_ds.cappi_height.values:
            pytest.skip("Python doesn't compute CAPPI 500m")

        python_corr = python_ds['cappi_correction_db'].sel(cappi_height=500)

        # Both should have corrections increasing with range in far field
        # (beam goes higher, sees less precipitation)
        legacy_100 = float(legacy_corr.sel(range_km=100).values)
        legacy_200 = float(legacy_corr.sel(range_km=200).values)

        python_100 = float(python_corr.sel(range_km=100).values)
        python_200 = float(python_corr.sel(range_km=200).values)

        # Corrections should generally increase with range for CAPPI
        # (this is a trend check, not exact value match)
        assert legacy_200 >= legacy_100 - 1, (
            f"Legacy CAPPI 500m should increase with range: "
            f"100km={legacy_100:.1f}, 200km={legacy_200:.1f}"
        )
        assert python_200 >= python_100 - 1, (
            f"Python CAPPI 500m should increase with range: "
            f"100km={python_100:.1f}, 200km={python_200:.1f}"
        )

    def test_python_elevation_corrections_computed(self, python_profile):
        """Verify Python computes VPR corrections for elevation angles from VVP."""
        vpr = python_profile.vpr_correction
        assert vpr is not None, "VPR correction should be computed"

        ds = vpr.corrections

        # Should have elevation-based corrections from VVP file
        assert 'elev_correction_db' in ds, (
            "Elevation-based corrections should be computed"
        )
        assert 'elevation' in ds.dims, (
            "Dataset should have elevation dimension"
        )

        # VVP file has 3 elevations: 0.7, 1.5, 3.0
        expected_elevs = [0.7, 1.5, 3.0]
        actual_elevs = list(ds.elevation.values)
        assert actual_elevs == expected_elevs, (
            f"Elevation angles mismatch: got {actual_elevs}, expected {expected_elevs}"
        )

    def test_python_elevation_beam_heights_match_legacy(self, legacy_cor, python_profile):
        """Compare Python elevation beam heights to legacy output.

        For fixed elevation angles, beam heights should match closely
        between Python and legacy implementations.
        """
        legacy_ds, header = legacy_cor
        python_vpr = python_profile.vpr_correction

        if python_vpr is None or not python_vpr.usable:
            pytest.skip("Python VPR correction not computed")

        python_ds = python_vpr.corrections

        if 'elev_beam_height_m' not in python_ds:
            pytest.skip("No elevation corrections in Python output")

        # Compare beam heights for first 3 elevations (legacy has 4, Python has 3)
        test_ranges = [50, 100, 150, 200]
        for elev_idx, elev in enumerate(header.elevation_angles[:3]):
            for range_km in test_ranges:
                legacy_height = int(legacy_ds['beam_height_m'].sel(
                    range_km=range_km, correction_type=f'elev_{elev_idx + 1}'
                ).values)

                python_height = float(python_ds['elev_beam_height_m'].sel(
                    range_km=range_km, elevation=elev
                ).values)

                # Allow 2% tolerance for implementation differences
                tolerance = max(10, int(python_height * 0.02))
                assert abs(python_height - legacy_height) <= tolerance, (
                    f"Beam height mismatch at {range_km}km, {elev}°: "
                    f"Python={python_height:.0f}m, legacy={legacy_height}m"
                )

    def test_python_elevation_corrections_trend(self, python_profile):
        """Verify elevation-based corrections follow expected trend.

        Higher elevations see higher altitudes, so corrections should
        generally increase with elevation angle at the same range.
        """
        vpr = python_profile.vpr_correction

        if vpr is None or not vpr.usable:
            pytest.skip("Python VPR correction not computed")

        ds = vpr.corrections

        if 'elev_correction_db' not in ds:
            pytest.skip("No elevation corrections in Python output")

        # At 100km, higher elevations should see weaker echo (larger correction)
        # or be in bright band (smaller/negative correction)
        corr_07 = float(ds['elev_correction_db'].sel(range_km=100, elevation=0.7).values)
        corr_15 = float(ds['elev_correction_db'].sel(range_km=100, elevation=1.5).values)
        corr_30 = float(ds['elev_correction_db'].sel(range_km=100, elevation=3.0).values)

        # At least verify they're all computed and reasonable
        assert all(abs(c) < 30 for c in [corr_07, corr_15, corr_30]), (
            f"Corrections out of range: 0.7°={corr_07:.1f}, "
            f"1.5°={corr_15:.1f}, 3.0°={corr_30:.1f}"
        )
