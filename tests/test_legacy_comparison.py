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
