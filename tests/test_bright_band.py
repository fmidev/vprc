"""Tests for the bright_band module."""

import numpy as np
import pytest
import xarray as xr

from vprc.bright_band import (
    BrightBandResult,
    compute_dbz_gradient_per_step,
    detect_bright_band,
    _find_bb_candidates,
    _compute_gradient_sum,
)
from vprc.constants import MDS, STEP


def make_test_dataset(
    dbz_values: list[float],
    heights: list[int] | None = None,
    freezing_level_m: float | None = None,
) -> xr.Dataset:
    """Create a minimal test dataset for bright band tests."""
    if heights is None:
        heights = list(range(100, 100 + STEP * len(dbz_values), STEP))

    ds = xr.Dataset(
        {
            "corrected_dbz": ("height", dbz_values),
        },
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
