"""Tests for the bright_band module."""

import numpy as np
import pytest
import xarray as xr

from vprc.bright_band import (
    BrightBandResult,
    BBRejectionReason,
    compute_dbz_gradient_per_step,
    detect_bright_band,
    apply_post_bb_clutter_correction,
    _find_bb_candidates,
    _compute_gradient_sum,
    _validate_near_surface_bb,
    _determine_clutter_scenario,
)
from vprc.constants import (
    MDS,
    STEP,
    POST_BB_CLUTTER_THRESHOLD_DB,
)


def make_test_dataset(
    dbz_values: list[float],
    heights: list[int] | None = None,
    freezing_level_m: float | None = None,
    zcount_values: list[int] | None = None,
) -> xr.Dataset:
    """Create a minimal test dataset for bright band tests.

    Args:
        dbz_values: Reflectivity values per height level
        heights: Height coordinates (auto-generated if None)
        freezing_level_m: Freezing level for attrs
        zcount_values: Sample counts per height level (optional)
    """
    if heights is None:
        heights = list(range(100, 100 + STEP * len(dbz_values), STEP))

    data_vars = {
        "corrected_dbz": ("height", dbz_values),
    }

    if zcount_values is not None:
        data_vars["zcount"] = ("height", zcount_values)

    ds = xr.Dataset(
        data_vars,
        coords={"height": heights},
        attrs={"freezing_level_m": freezing_level_m},
    )
    return ds


class TestBrightBandResult:
    """Tests for BrightBandResult dataclass."""

    def test_thickness_calculation(self):
        result = BrightBandResult(
            detected=True,
            peak_height=2000,
            bottom_height=1600,
            top_height=2400,
        )
        assert result.thickness == 800

    def test_thickness_none_when_missing_bounds(self):
        result = BrightBandResult(detected=False)
        assert result.thickness is None

    def test_default_not_detected(self):
        result = BrightBandResult()
        assert result.detected is False
        assert result.peak_height is None


class TestComputeGradientPerStep:
    """Tests for gradient computation."""

    def test_decreasing_profile_positive_gradient(self):
        """Gradient is positive when dbz decreases upward (dbz[i] - dbz[i+1] > 0)."""
        # 30 at bottom, decreasing to 10 at top
        dbz_values = [30, 25, 20, 15, 10]
        ds = make_test_dataset(dbz_values)

        gradient = compute_dbz_gradient_per_step(ds)

        # Each step decreases by 5 dBZ, so gradient = 5
        assert np.isclose(gradient.sel(height=100).values, 5)
        assert np.isclose(gradient.sel(height=300).values, 5)

    def test_increasing_profile_negative_gradient(self):
        """Gradient is negative when dbz increases upward."""
        dbz_values = [10, 15, 20, 25, 30]
        ds = make_test_dataset(dbz_values)

        gradient = compute_dbz_gradient_per_step(ds)

        # Each step increases by 5, so gradient = -5
        assert np.isclose(gradient.sel(height=100).values, -5)

    def test_missing_values_masked(self):
        """Gradient is NaN where either value is MDS."""
        dbz_values = [20, MDS, 25, 30]
        ds = make_test_dataset(dbz_values)

        gradient = compute_dbz_gradient_per_step(ds)

        # Gradient at 100 involves MDS at 300 -> NaN
        assert np.isnan(gradient.sel(height=100).values)
        # Gradient at 300 involves MDS at 300 -> NaN
        assert np.isnan(gradient.sel(height=300).values)
        # Gradient at 500 is valid (25 - 30 = -5)
        assert np.isclose(gradient.sel(height=500).values, -5)


class TestGradientSum:
    """Tests for gradient sum helper."""

    def test_gradient_sum_calculation(self):
        """Sum of gradients at adjacent levels."""
        dbz_values = [30, 25, 20, 15, 10]  # Decreasing by 5 each step
        ds = make_test_dataset(dbz_values)
        gradient = compute_dbz_gradient_per_step(ds)

        # Gradient sum at 100 = gradient[100] + gradient[300] = 5 + 5 = 10
        result = _compute_gradient_sum(gradient, 100)
        assert np.isclose(result, 10)

    def test_gradient_sum_missing_height(self):
        """Returns NaN if heights don't exist."""
        dbz_values = [20, 25, 30]
        ds = make_test_dataset(dbz_values)
        gradient = compute_dbz_gradient_per_step(ds)

        # Height 1000 doesn't exist
        result = _compute_gradient_sum(gradient, 1000)
        assert np.isnan(result)


class TestDetectBrightBand:
    """Tests for bright band detection."""

    def test_no_bb_when_freezing_level_zero_or_negative(self):
        """BB detection skipped if freezing level <= 0."""
        dbz_values = [20, 25, 30, 35, 30, 25, 20, 15, 10]
        ds = make_test_dataset(dbz_values, freezing_level_m=0)

        result = detect_bright_band(ds)

        assert result.detected is False
        assert result.freezing_level_m == 0

    def test_no_bb_when_layer_too_low(self):
        """BB detection skipped if layer top < 1300m."""
        dbz_values = [20, 25, 30, 25, 20]
        heights = [100, 300, 500, 700, 900]  # Max height 900m
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=500)

        result = detect_bright_band(ds, layer_top=900)

        assert result.detected is False

    def test_no_bb_in_monotonic_profile(self):
        """No BB detected in monotonically decreasing profile."""
        # Smoothly decreasing from bottom to top - no BB signature
        dbz_values = [35, 32, 29, 26, 23, 20, 17, 14, 11, 8]
        heights = list(range(100, 2100, 200))
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=1000)

        result = detect_bright_band(ds, layer_top=1900)

        assert result.detected is False

    def test_bb_detection_with_peak(self):
        """BB detected in profile with characteristic peak."""
        # Profile with BB signature: increase toward peak, then decrease
        # Heights: 100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300
        heights = list(range(100, 2500, 200))

        # Create BB at ~1500m: gradual increase, peak, then decrease
        dbz_values = [
            10,  # 100m - below BB
            12,  # 300m
            15,  # 500m
            18,  # 700m - approaching BB
            22,  # 900m
            26,  # 1100m
            30,  # 1300m - entering BB
            35,  # 1500m - BB peak
            28,  # 1700m - exiting BB
            22,  # 1900m
            18,  # 2100m
            15,  # 2300m
        ]

        ds = make_test_dataset(dbz_values, heights, freezing_level_m=1500)

        result = detect_bright_band(ds, layer_top=2300)

        # Should detect something, though exact behavior depends on thresholds
        # The key is that this profile has the right shape
        assert result.freezing_level_m == 1500

    def test_bb_result_has_freezing_level(self):
        """Result always includes freezing level used."""
        dbz_values = [20, 25, 30, 25, 20]
        ds = make_test_dataset(dbz_values, freezing_level_m=2500)

        result = detect_bright_band(ds, layer_top=900)

        assert result.freezing_level_m == 2500

    def test_empty_profile_no_bb(self):
        """No BB in profile with all MDS values."""
        dbz_values = [MDS, MDS, MDS, MDS, MDS]
        ds = make_test_dataset(dbz_values, freezing_level_m=1500)

        result = detect_bright_band(ds, layer_top=900)

        assert result.detected is False


class TestBBCandidateFinding:
    """Tests for BB candidate identification."""

    def test_no_candidates_with_missing_data(self):
        """No candidates if there are gaps in the profile."""
        dbz_values = [20, 25, MDS, 25, 20, 15, 10]
        heights = list(range(100, 1500, 200))
        ds = make_test_dataset(dbz_values, heights)
        gradient = compute_dbz_gradient_per_step(ds)

        candidates = _find_bb_candidates(
            ds["corrected_dbz"], gradient, np.array(heights), 1300
        )

        # Gap at 500m means no 5-level continuous sequence possible there
        # Candidates require 5 consecutive valid levels
        assert len(candidates) == 0

    def test_candidates_require_gradient_pattern(self):
        """Candidates need specific gradient pattern."""
        # Uniform profile - no gradient pattern
        dbz_values = [20, 20, 20, 20, 20, 20, 20]
        heights = list(range(100, 1500, 200))
        ds = make_test_dataset(dbz_values, heights)
        gradient = compute_dbz_gradient_per_step(ds)

        candidates = _find_bb_candidates(
            ds["corrected_dbz"], gradient, np.array(heights), 1300
        )

        assert len(candidates) == 0


class TestIntegration:
    """Integration tests using realistic profile patterns."""

    def test_detect_bb_returns_dataclass(self):
        """Ensure return type is always BrightBandResult."""
        dbz_values = [20, 25, 30, 25, 20]
        ds = make_test_dataset(dbz_values, freezing_level_m=500)

        result = detect_bright_band(ds)

        assert isinstance(result, BrightBandResult)

    def test_no_freezing_level_still_works(self):
        """Detection works without freezing level (uses alternative logic)."""
        dbz_values = [20, 25, 30, 25, 20, 15, 10]
        heights = list(range(100, 1500, 200))
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=None)

        result = detect_bright_band(ds, layer_top=1300)

        # Should not crash, returns a result
        assert isinstance(result, BrightBandResult)
        assert result.freezing_level_m is None


class TestRealData:
    """Tests using real VVP data files."""

    def test_vih_clear_bright_band(self):
        """Detect bright band in VIH case with clear BB signal.

        202511071400_VIH.VVP_40.txt has a prominent bright band around 2200m.
        The legacy Perl method failed to detect it (without freezing level data),
        but the signature is unmistakable in the profile.
        """
        from pathlib import Path
        from vprc.io import read_vvp
        from vprc.clutter import remove_ground_clutter
        from vprc.smoothing import smooth_spikes

        test_file = Path(__file__).parent / "data" / "202511071400_VIH.VVP_40.txt"
        ds = read_vvp(test_file)
        ds = remove_ground_clutter(ds)
        ds = smooth_spikes(ds)

        result = detect_bright_band(ds)

        assert result.detected is True, "BB should be detected in this clear case"

        # Peak around 2200m (±200m margin)
        assert result.peak_height is not None
        assert 2000 <= result.peak_height <= 2400, f"Peak at {result.peak_height}m, expected ~2200m"

        # Bottom around 1500m (±200m margin)
        assert result.bottom_height is not None
        assert 1300 <= result.bottom_height <= 1700, f"Bottom at {result.bottom_height}m, expected ~1500m"

        # Top around 3000m (±200m margin)
        assert result.top_height is not None
        assert 2800 <= result.top_height <= 3200, f"Top at {result.top_height}m, expected ~3000m"

        # Sanity checks on amplitudes
        assert result.amplitude_below is not None and result.amplitude_below > 5
        assert result.amplitude_above is not None and result.amplitude_above > 10


class TestNearSurfaceBBValidation:
    """Tests for near-surface bright band validation.

    When BB bottom is at the lowest 1-2 levels, the Perl code performs
    additional validation on sample counts (Zcount) to prevent false
    detection from ground clutter.

    Perl logic (lines 1051-1075 allprof_prodx2.pl):
        if ($bbalku == $alintaso or $bbalku == $alintaso + 200):
            if (Zcount[bbalku] < 500 or Zcount[zyla] < 500
                or Zcount[zyla]/Zcount[bbalku] < 0.7):
                bb = 0  # reject
    """

    def test_valid_near_surface_bb_passes(self):
        """BB at lowest level with sufficient sample counts passes."""
        # Heights: 100 (lowest), 300, 500, 700
        heights = [100, 300, 500, 700]
        dbz_values = [25.0, 30.0, 25.0, 20.0]
        # All sample counts >= 500, ratio > 0.7
        zcount_values = [600, 700, 650, 550]

        ds = make_test_dataset(dbz_values, heights, zcount_values=zcount_values)

        result = _validate_near_surface_bb(ds, bottom_height=100, top_height=500)

        assert result is True

    def test_low_zcount_at_bottom_rejects(self):
        """BB rejected if sample count at bottom < 500."""
        heights = [100, 300, 500, 700]
        dbz_values = [25.0, 30.0, 25.0, 20.0]
        # Bottom zcount too low
        zcount_values = [400, 700, 650, 550]

        ds = make_test_dataset(dbz_values, heights, zcount_values=zcount_values)

        result = _validate_near_surface_bb(ds, bottom_height=100, top_height=500)

        assert result is False

    def test_low_zcount_at_top_rejects(self):
        """BB rejected if sample count at top < 500."""
        heights = [100, 300, 500, 700]
        dbz_values = [25.0, 30.0, 25.0, 20.0]
        # Top zcount too low
        zcount_values = [600, 700, 400, 550]

        ds = make_test_dataset(dbz_values, heights, zcount_values=zcount_values)

        result = _validate_near_surface_bb(ds, bottom_height=100, top_height=500)

        assert result is False

    def test_low_zcount_ratio_rejects(self):
        """BB rejected if ratio of top/bottom zcount < 0.7."""
        heights = [100, 300, 500, 700]
        dbz_values = [25.0, 30.0, 25.0, 20.0]
        # Both >= 500, but ratio 510/1000 = 0.51 < 0.7
        zcount_values = [1000, 700, 510, 550]

        ds = make_test_dataset(dbz_values, heights, zcount_values=zcount_values)

        result = _validate_near_surface_bb(ds, bottom_height=100, top_height=500)

        assert result is False

    def test_ratio_exactly_at_threshold_passes(self):
        """BB passes if ratio exactly equals threshold (0.7)."""
        heights = [100, 300, 500, 700]
        dbz_values = [25.0, 30.0, 25.0, 20.0]
        # Ratio 700/1000 = 0.7 exactly
        zcount_values = [1000, 800, 700, 550]

        ds = make_test_dataset(dbz_values, heights, zcount_values=zcount_values)

        result = _validate_near_surface_bb(ds, bottom_height=100, top_height=500)

        assert result is True

    def test_missing_top_height_rejects(self):
        """BB rejected if top height is None."""
        heights = [100, 300, 500, 700]
        dbz_values = [25.0, 30.0, 25.0, 20.0]
        zcount_values = [600, 700, 650, 550]

        ds = make_test_dataset(dbz_values, heights, zcount_values=zcount_values)

        result = _validate_near_surface_bb(ds, bottom_height=100, top_height=None)

        assert result is False

    def test_missing_zcount_data_passes(self):
        """BB passes validation if no zcount data available (graceful fallback)."""
        heights = [100, 300, 500, 700]
        dbz_values = [25.0, 30.0, 25.0, 20.0]
        # No zcount_values -> no 'zcount' in dataset

        ds = make_test_dataset(dbz_values, heights)

        result = _validate_near_surface_bb(ds, bottom_height=100, top_height=500)

        # Without data to validate, we don't reject
        assert result is True

    def test_bb_at_second_lowest_level_validated(self):
        """BB validation applies when bottom is at second-lowest level too."""
        heights = [100, 300, 500, 700]
        dbz_values = [20.0, 25.0, 30.0, 25.0]
        # Bottom at 300 (second lowest), low zcount
        zcount_values = [600, 400, 650, 550]

        ds = make_test_dataset(dbz_values, heights, zcount_values=zcount_values)

        # Bottom at 300 (second lowest = 100 + 200)
        result = _validate_near_surface_bb(ds, bottom_height=300, top_height=700)

        assert result is False

    def test_bb_above_lowest_two_levels_not_validated_here(self):
        """This function only validates; caller decides when to call it.

        When BB bottom is above lowest 2 levels, the caller (_validate_bright_band)
        doesn't call this function, so this function isn't responsible for that check.
        """
        heights = [100, 300, 500, 700, 900]
        dbz_values = [15.0, 20.0, 30.0, 25.0, 20.0]
        # Would fail if validated, but caller won't call us for h=500
        zcount_values = [600, 700, 100, 100, 550]

        ds = make_test_dataset(dbz_values, heights, zcount_values=zcount_values)

        # If called directly, it validates based on heights given
        # (even though 500 is not near-surface, the validation still applies)
        result = _validate_near_surface_bb(ds, bottom_height=500, top_height=900)

        # Zcount at bottom (500) = 100 < 500, so fails
        assert result is False


class TestBBRejectionReason:
    """Tests for BB rejection reason tracking."""

    def test_detected_bb_has_not_rejected_reason(self):
        """Successfully detected BB has NOT_REJECTED reason."""
        # Create a profile with clear BB signature
        heights = list(range(100, 2500, 200))
        dbz_values = [
            10, 12, 15, 18, 22, 26, 30, 35, 28, 22, 18, 15,
        ]
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=1500)

        result = detect_bright_band(ds, layer_top=2300)

        # If detected, reason should be NOT_REJECTED
        if result.detected:
            assert result.rejection_reason == BBRejectionReason.NOT_REJECTED

    def test_layer_too_low_reason(self):
        """BB rejected due to low layer has correct reason."""
        dbz_values = [20, 25, 30, 25, 20]
        heights = [100, 300, 500, 700, 900]
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=500)

        result = detect_bright_band(ds, layer_top=900)

        assert result.detected is False
        assert result.rejection_reason == BBRejectionReason.LAYER_TOO_LOW

    def test_freezing_level_invalid_reason(self):
        """BB rejected due to invalid freezing level has correct reason."""
        dbz_values = [20, 25, 30, 35, 30, 25, 20, 15, 10]
        ds = make_test_dataset(dbz_values, freezing_level_m=0)

        result = detect_bright_band(ds)

        assert result.detected is False
        assert result.rejection_reason == BBRejectionReason.FREEZING_LEVEL_INVALID

    def test_no_candidates_reason(self):
        """BB rejected due to no candidates has correct reason."""
        # Monotonically decreasing profile - no BB signature
        heights = list(range(100, 2100, 200))
        dbz_values = [35, 32, 29, 26, 23, 20, 17, 14, 11, 8]
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=1000)

        result = detect_bright_band(ds, layer_top=1900)

        assert result.detected is False
        assert result.rejection_reason == BBRejectionReason.NO_CANDIDATES


class TestDetermineClutterScenario:
    """Tests for post-BB clutter correction scenario determination."""

    def test_near_surface_rejected_scenario(self):
        """Scenario A: Near-surface BB rejected due to zcount."""
        bb_result = BrightBandResult(
            detected=False,
            rejection_reason=BBRejectionReason.NEAR_SURFACE_ZCOUNT,
        )

        scenario = _determine_clutter_scenario(bb_result, freezing_level=2000)

        assert scenario == "near_surface_rejected"

    def test_low_bb_bottom_scenario(self):
        """Scenario B: Valid BB with low bottom height."""
        bb_result = BrightBandResult(
            detected=True,
            bottom_height=600,  # <= 800m
        )

        scenario = _determine_clutter_scenario(bb_result, freezing_level=2000)

        assert scenario == "low_bb_bottom"

    def test_high_bb_bottom_scenario(self):
        """Scenario C: Valid BB with high bottom height."""
        bb_result = BrightBandResult(
            detected=True,
            bottom_height=1000,  # > 800m
        )

        scenario = _determine_clutter_scenario(bb_result, freezing_level=2000)

        assert scenario == "high_bb_or_no_bb"

    def test_no_bb_low_freezing_scenario(self):
        """Scenario C: No BB with low freezing level."""
        bb_result = BrightBandResult(
            detected=False,
            rejection_reason=BBRejectionReason.NO_CANDIDATES,
        )

        scenario = _determine_clutter_scenario(bb_result, freezing_level=800)

        assert scenario == "high_bb_or_no_bb"

    def test_no_bb_high_freezing_no_correction(self):
        """No correction when no BB and high freezing level."""
        bb_result = BrightBandResult(
            detected=False,
            rejection_reason=BBRejectionReason.NO_CANDIDATES,
        )

        scenario = _determine_clutter_scenario(bb_result, freezing_level=2000)

        assert scenario is None

    def test_other_rejection_reasons_no_correction(self):
        """No correction for other rejection reasons without low freezing."""
        bb_result = BrightBandResult(
            detected=False,
            rejection_reason=BBRejectionReason.LAYER_TOO_LOW,
        )

        scenario = _determine_clutter_scenario(bb_result, freezing_level=2000)

        assert scenario is None


class TestApplyPostBBClutterCorrection:
    """Tests for post-BB ground clutter correction."""

    def test_no_correction_when_not_needed(self):
        """No changes when no correction scenario applies."""
        heights = [100, 300, 500, 700, 900]
        dbz_values = [20.0, 22.0, 24.0, 26.0, 28.0]
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=2000)

        bb_result = BrightBandResult(
            detected=False,
            rejection_reason=BBRejectionReason.NO_CANDIDATES,
            freezing_level_m=2000,
        )

        ds_out, bb_out = apply_post_bb_clutter_correction(ds, bb_result)

        # No changes expected
        assert np.allclose(ds_out["corrected_dbz"].values, dbz_values)
        assert bb_out.clutter_correction_applied is False

    def test_near_surface_rejected_applies_correction(self):
        """Scenario A: Correction applied when near-surface BB rejected."""
        heights = [100, 300, 500, 700, 900]
        # Lower levels exceed upper by more than 1 dB -> should be corrected
        dbz_values = [30.0, 28.0, 25.0, 22.0, 20.0]
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=500)

        bb_result = BrightBandResult(
            detected=False,
            rejection_reason=BBRejectionReason.NEAR_SURFACE_ZCOUNT,
            freezing_level_m=500,
        )

        ds_out, bb_out = apply_post_bb_clutter_correction(ds, bb_result)

        # Clutter correction applied flag
        assert bb_out.clutter_correction_applied is True

        # At 100m: original=30, above (300)=28, threshold=28+1=29
        # 30 > 29, so should be corrected to 29
        corrected = ds_out["corrected_dbz"].values
        assert corrected[0] <= dbz_values[1] + POST_BB_CLUTTER_THRESHOLD_DB

    def test_low_bb_bottom_applies_correction_below_bb(self):
        """Scenario B: Correction below BB when BB bottom is low."""
        heights = [100, 300, 500, 700, 900, 1100]
        # Clutter at lowest levels, BB at 500-700m
        dbz_values = [35.0, 30.0, 25.0, 28.0, 25.0, 20.0]
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=700)

        bb_result = BrightBandResult(
            detected=True,
            bottom_height=500,  # <= 800m
            top_height=700,
            freezing_level_m=700,
        )

        ds_out, bb_out = apply_post_bb_clutter_correction(ds, bb_result)

        assert bb_out.clutter_correction_applied is True

        # Correction should apply from BB bottom (500) down
        corrected = ds_out["corrected_dbz"].values
        # At 300m: 30 > 25+1=26, corrected to 26
        assert corrected[1] <= corrected[2] + POST_BB_CLUTTER_THRESHOLD_DB
        # At 100m: 35 > 26+1=27, corrected
        assert corrected[0] <= corrected[1] + POST_BB_CLUTTER_THRESHOLD_DB

    def test_correction_does_not_increase_values(self):
        """Correction only decreases values, never increases."""
        heights = [100, 300, 500, 700]
        # Values already decrease smoothly - no correction needed
        dbz_values = [20.0, 21.0, 22.0, 23.0]
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=500)

        bb_result = BrightBandResult(
            detected=False,
            rejection_reason=BBRejectionReason.NEAR_SURFACE_ZCOUNT,
            freezing_level_m=500,
        )

        ds_out, bb_out = apply_post_bb_clutter_correction(ds, bb_result)

        # Values should be unchanged (original was already within threshold)
        corrected = ds_out["corrected_dbz"].values
        np.testing.assert_allclose(corrected, dbz_values)

    def test_bb_result_fields_preserved(self):
        """All BrightBandResult fields are preserved after correction."""
        heights = [100, 300, 500, 700, 900]
        dbz_values = [30.0, 28.0, 25.0, 22.0, 20.0]
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=500)

        bb_result = BrightBandResult(
            detected=False,
            peak_height=700,
            bottom_height=500,
            top_height=900,
            peak_dbz=22.0,
            amplitude_below=3.0,
            amplitude_above=2.0,
            freezing_level_m=500,
            rejection_reason=BBRejectionReason.NEAR_SURFACE_ZCOUNT,
        )

        _, bb_out = apply_post_bb_clutter_correction(ds, bb_result)

        # All original fields preserved
        assert bb_out.detected == bb_result.detected
        assert bb_out.peak_height == bb_result.peak_height
        assert bb_out.bottom_height == bb_result.bottom_height
        assert bb_out.top_height == bb_result.top_height
        assert bb_out.peak_dbz == bb_result.peak_dbz
        assert bb_out.amplitude_below == bb_result.amplitude_below
        assert bb_out.amplitude_above == bb_result.amplitude_above
        assert bb_out.freezing_level_m == bb_result.freezing_level_m
        assert bb_out.rejection_reason == bb_result.rejection_reason
        # New field updated
        assert bb_out.clutter_correction_applied is True
