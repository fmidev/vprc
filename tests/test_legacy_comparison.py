"""End-to-end tests comparing Python implementation against legacy Perl output.

These tests validate that the Python reimplementation produces results
matching the reference Perl implementation (allprof_prodx2.pl).

Test data: tests/data/202508241100_KAN.VVP_40.* files
- .txt: Input VVP profile
- .profile: Expected corrected profile output

Height coordinate system:
Both Python and legacy use heights above antenna (meters). The legacy code
converts sea-level heights from VVP files using:
    height_above_antenna = height_sea_level - antenna_height_m
Levels below antenna are dropped (negative heights).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vprc import PrecipitationType, get_radar_metadata, process_vvp
from vprc.constants import MDS

from .legacy_parser import parse_legacy_profile


# Paths to test data
DATA_DIR = Path(__file__).parent / 'data'
INPUT_FILE = DATA_DIR / '202508241100_KAN.VVP_40.txt'
EXPECTED_PROFILE = DATA_DIR / '202508241100_KAN.VVP_40.profile'

@pytest.fixture
def expected_profile():
    """Load the expected legacy profile output."""
    ds, header, footer = parse_legacy_profile(EXPECTED_PROFILE)
    return ds, header, footer


@pytest.fixture
def processed_result(expected_profile):
    """Run with the reference freezing height, converted to the API's ASL."""
    _, _, footer = expected_profile
    antenna_height = get_radar_metadata("KAN")["antenna_height_m"]
    return process_vvp(
        INPUT_FILE,
        freezing_level_m=footer.freezing_level + antenna_height,
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

        np.testing.assert_array_equal(
            result_ds.height.values,
            expected_ds.height.values[:len(result_ds.height)],
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

        # Perl prints a fixed 10 km grid; the reader stops at the input top.
        actual = result_ds['lin_dbz'].reindex(height=expected_ds.height, fill_value=MDS)
        np.testing.assert_allclose(
            actual.values,
            expected_ds['lin_dbz'].values,
            rtol=0.01,
            err_msg="lin_dbz values do not match"
        )

    def test_corrected_dbz_values_match(self, processed_result, expected_profile):
        """Verify corrected_dbz values match after full pipeline."""
        expected_ds, _, _ = expected_profile
        result_ds = processed_result.dataset

        actual = result_ds['corrected_dbz'].reindex(
            height=expected_ds.height, fill_value=MDS
        )
        np.testing.assert_allclose(
            actual.values,
            expected_ds['corrected_dbz'].values,
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

        layer = classification.lowest_layer
        assert layer is not None
        assert layer.bottom_height == footer.bottom_height, (
            f"Bottom height mismatch: got {layer.bottom_height}, "
            f"expected {footer.bottom_height}"
        )
        assert layer.top_height == footer.top_height, (
            f"Top height mismatch: got {layer.top_height}, "
            f"expected {footer.top_height}"
        )

    def test_evaporation_matches(self, processed_result, expected_profile):
        """Compare the actual evaporation column, not a BB amplitude."""
        _, _, footer = expected_profile
        layer = processed_result.classification.lowest_layer
        assert layer is not None
        assert layer.evaporation_amount == pytest.approx(footer.evaporation, abs=0.1)

    def test_reflectivity_sample_counts_match(self, processed_result, expected_profile):
        """The legacy oracle contains raw ZCnt, not the IRIS Count column."""
        expected_ds, _, _ = expected_profile
        actual = processed_result.dataset["zcount"].reindex(
            height=expected_ds.height, fill_value=0
        )
        np.testing.assert_array_equal(actual, expected_ds["zcount"])

    def test_ground_clutter_classification_matches(self, processed_result, expected_profile):
        _, _, footer = expected_profile
        layer = processed_result.classification.lowest_layer
        assert layer is not None
        assert layer.has_clutter == bool(footer.ground_clutter)

    def test_precipitation_classification_matches(self, processed_result, expected_profile):
        """Verify precipitation type classification matches legacy.

        Legacy values: 0=none, 1=untyped, 2=snow, 3=sleet, 4=rain.
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
        expected_phase = {
            0: PrecipitationType.UNKNOWN,
            1: PrecipitationType.UNKNOWN,
            2: PrecipitationType.SNOW,
            3: PrecipitationType.SLEET,
            4: PrecipitationType.RAIN,
        }[footer.precipitation]
        assert classification.precipitation_type == expected_phase

    def test_quality_weight_matches(self, processed_result, expected_profile):
        """Compare quality at the reference's two-decimal output precision."""
        _, _, footer = expected_profile
        assert processed_result.vpr_correction is not None
        assert processed_result.vpr_correction.quality_weight == pytest.approx(
            footer.quality_weight, abs=0.005
        )


class TestBrightBandComparison:
    """Compare bright band detection using the same freezing coordinate."""

    def test_bright_band_detection_agrees(self, processed_result, expected_profile):
        """Verify bright band detection matches legacy.

        Legacy BB flag: 0=not detected, 1=detected

        """
        _, _, footer = expected_profile
        bb_result = processed_result.bright_band

        legacy_bb_detected = footer.bright_band == 1

        assert bb_result.detected == legacy_bb_detected, (
            f"Bright band detection mismatch: "
            f"legacy={legacy_bb_detected}, python={bb_result.detected}"
        )

    def test_bright_band_height_matches(self, processed_result, expected_profile):
        """Verify bright band height matches legacy (if detected)."""
        _, _, footer = expected_profile
        bb_result = processed_result.bright_band

        assert bb_result.detected == bool(footer.bright_band)
        if footer.bright_band == 1:
            # The footer prints weighted bbhuippu, not the discrete bbalku.
            assert bb_result.refined_peak_height is not None
            assert abs(bb_result.refined_peak_height - footer.bb_height) <= 200, (
                f"BB height mismatch: got {bb_result.refined_peak_height}, "
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
        assert processed_result.classification.usable_for_vpr
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
    def python_profile(self, processed_result, legacy_cor, expected_profile):
        """Use the reference's range and minimum elevation, not API defaults."""
        from vprc.vpr_correction import compute_vpr_correction

        legacy_ds, header = legacy_cor
        _, _, footer = expected_profile
        correction = compute_vpr_correction(
            processed_result.dataset,
            freezing_level_m=header.freezing_level_m,
            max_range_km=int(legacy_ds.range_km.max()),
            min_elevation_deg=header.elevation_angles[0],
            elevation_angles_deg=tuple(header.elevation_angles),
            is_precipitation=header.precip_code > 0,
            layer_top_height_m=footer.top_height,
            horizon_angle_deg=0,
            bb_result=processed_result.bright_band,
        )
        return replace(processed_result, vpr_correction=correction)

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
        assert header.precip_code == 2  # snow
        assert header.elevation_angles == [0.7, 1.5, 3.0, 3.3]

    @pytest.mark.parametrize("cappi_height", [500, 1000])
    def test_cappi_beam_height_calculation(
        self, legacy_cor, python_profile, cappi_height
    ):
        """Compare pseudo-CAPPI heights above antenna over the full range."""
        legacy_ds, _ = legacy_cor
        assert python_profile.vpr_correction is not None
        actual = python_profile.vpr_correction.corrections["cappi_beam_height_m"].sel(
            cappi_height=cappi_height
        )
        expected = legacy_ds["beam_height_m"].sel(
            correction_type=f"cappi_{cappi_height}"
        )
        # Same Rinehart/wradlib geometry allowance as fixed elevations below.
        tolerance = np.maximum(10, np.floor(actual.values * 0.02))
        assert np.all(np.abs(actual.values - expected.values) <= tolerance)

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

    @pytest.mark.parametrize("cappi_height", [500, 1000])
    @pytest.mark.parametrize(
        ("python_variable", "legacy_variable"),
        [
            ("cappi_correction_db", "correction_rain_db"),
            ("cappi_clim_correction_db", "correction_klim_db"),
        ],
    )
    def test_python_cappi_corrections_match(
        self, legacy_cor, python_profile, cappi_height, python_variable, legacy_variable
    ):
        """Compare instantaneous and climatological products separately."""
        legacy_ds, _ = legacy_cor
        python_vpr = python_profile.vpr_correction
        assert python_vpr is not None and python_vpr.usable
        python_ds = python_vpr.corrections
        np.testing.assert_array_equal(python_ds.range_km, legacy_ds.range_km)
        np.testing.assert_allclose(
            python_ds[python_variable].sel(cappi_height=cappi_height),
            legacy_ds[legacy_variable].sel(correction_type=f"cappi_{cappi_height}"),
            rtol=0,
            atol=0.1,
        )

    def test_python_elevation_corrections_computed(self, processed_result):
        """Verify Python computes VPR corrections for elevation angles from VVP."""
        vpr = processed_result.vpr_correction
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

        assert python_vpr is not None and python_vpr.usable
        python_ds = python_vpr.corrections
        assert 'elev_beam_height_m' in python_ds

        # The comparison fixture requests all four reference elevations.
        test_ranges = [50, 100, 150, 200]
        for elev_idx, elev in enumerate(header.elevation_angles):
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

    @pytest.mark.parametrize(
        ("python_variable", "legacy_variable"),
        [
            ("elev_correction_db", "correction_rain_db"),
            ("elev_clim_correction_db", "correction_klim_db"),
        ],
    )
    def test_python_elevation_corrections_match(
        self, legacy_cor, python_profile, python_variable, legacy_variable
    ):
        """Compare each VVP elevation against its matching reference product."""
        legacy_ds, header = legacy_cor
        vpr = python_profile.vpr_correction
        assert vpr is not None and vpr.usable
        ds = vpr.corrections
        np.testing.assert_array_equal(ds.range_km, legacy_ds.range_km)
        for elevation in ds.elevation.values:
            reference_index = header.elevation_angles.index(float(elevation)) + 1
            np.testing.assert_allclose(
                ds[python_variable].sel(elevation=elevation),
                legacy_ds[legacy_variable].sel(
                    correction_type=f"elev_{reference_index}"
                ),
                rtol=0,
                atol=0.1,
            )


@pytest.mark.parametrize(
    ("stem", "radar_code"),
    [
        ("202208171100_VIM.VVP_40", "VIM"),
        ("202208171245_VIM.VVP_40", "VIM"),
        # Reference generated by running allprof_prodx2.pl on the input.
        ("202611112000_KES.VVP_40", "KES"),
    ],
)
def test_levels_absent_from_input_match_legacy_profile(stem, radar_code):
    """Profiles with levels absent from the VVP file keep the legacy grid.

    Perl stores levels by height, so absent levels are output as no-echo
    levels and may be filled by smoothing. Select rather than reindex the
    Python result so that a dropped level fails instead of matching MDS.
    """
    expected, _, footer = parse_legacy_profile(DATA_DIR / f"{stem}.profile")
    antenna_height = get_radar_metadata(radar_code)["antenna_height_m"]
    result = process_vvp(
        DATA_DIR / f"{stem}.txt",
        freezing_level_m=footer.freezing_level + antenna_height,
    ).dataset
    assert result.attrs["missing_levels_m"]
    heights = expected.height.where(expected.height <= result.height.max(), drop=True)
    for variable in ("lin_dbz", "corrected_dbz", "zcount"):
        np.testing.assert_allclose(
            result[variable].sel(height=heights).values,
            expected[variable].sel(height=heights).values,
            rtol=0.01,
            atol=0.1,
            err_msg=f"{variable} does not match legacy output for {stem}",
        )
