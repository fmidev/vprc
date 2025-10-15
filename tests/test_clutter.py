#!/usr/bin/env python3
"""
Tests for ground clutter detection and removal.
"""

import numpy as np
import pytest
import xarray as xr

from vprc.clutter import compute_gradient, remove_ground_clutter
from vprc.constants import MDS, GROUND_CLUTTER_GRADIENT_THRESHOLD, STEP, MIN_SAMPLES


class TestComputeGradient:
    """Tests for gradient computation with quality checking."""

    def test_gradient_simple_profile(self):
        """Test gradient calculation on a simple increasing profile."""
        # Create simple test dataset
        heights = np.array([100, 300, 500, 700, 900])
        dbz_values = np.array([10.0, 15.0, 20.0, 25.0, 30.0])  # +5 dBZ per 200m
        counts = np.array([100, 100, 100, 100, 100])

        ds = xr.Dataset(
            {
                'corrected_dbz': ('height', dbz_values),
                'count': ('height', counts),
            },
            coords={'height': heights}
        )

        gradient = compute_gradient(ds)

        # Expected gradient: 5 dBZ / 200 m = 0.025 dBZ/m
        expected_grad = 0.025

        # Check that gradients are approximately correct (within tolerance)
        # xarray uses centered differences for interior points
        assert gradient.sel(height=500).values == pytest.approx(expected_grad, abs=0.001)

    def test_gradient_with_low_sample_count(self):
        """Test that gradients with low sample counts are marked invalid."""
        heights = np.array([100, 300, 500, 700, 900])
        dbz_values = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
        counts = np.array([100, 20, 100, 100, 100])  # Second level has low count

        ds = xr.Dataset(
            {
                'corrected_dbz': ('height', dbz_values),
                'count': ('height', counts),
            },
            coords={'height': heights}
        )

        gradient = compute_gradient(ds, min_samples=MIN_SAMPLES)

        # Gradient at 100m involves 100m and 300m - 300m has low count
        assert np.isnan(gradient.sel(height=100).values)

        # Gradient at 300m involves 300m and 500m - 300m has low count
        assert np.isnan(gradient.sel(height=300).values)

        # Gradient at 500m involves 500m and neighbors - should be valid
        # (depends on xarray's centered difference implementation)

    def test_gradient_negative(self):
        """Test gradient calculation with decreasing profile (negative gradient)."""
        heights = np.array([100, 300, 500, 700])
        dbz_values = np.array([30.0, 28.0, 26.0, 24.0])  # -2 dBZ per 200m
        counts = np.array([100, 100, 100, 100])

        ds = xr.Dataset(
            {
                'corrected_dbz': ('height', dbz_values),
                'count': ('height', counts),
            },
            coords={'height': heights}
        )

        gradient = compute_gradient(ds)

        # Expected gradient: -2 dBZ / 200 m = -0.01 dBZ/m
        expected_grad = -0.01

        # Check interior points
        assert gradient.sel(height=300).values == pytest.approx(expected_grad, abs=0.001)


class TestRemoveGroundClutter:
    """Tests for ground clutter removal algorithm."""

    def test_no_clutter_profile_unchanged(self):
        """Test that clean profiles are not modified."""
        heights = np.array([100, 300, 500, 700, 900])
        dbz_values = np.array([20.0, 22.0, 24.0, 26.0, 28.0])  # Gradual increase
        counts = np.array([100, 100, 100, 100, 100])

        ds = xr.Dataset(
            {
                'corrected_dbz': ('height', dbz_values),
                'count': ('height', counts),
            },
            coords={'height': heights},
            attrs={'lowest_level_offset_m': 100, 'freezing_level_m': 2000}
        )

        ds_corrected = remove_ground_clutter(ds)

        # Profile should be unchanged (no steep negative gradients)
        np.testing.assert_array_almost_equal(
            ds_corrected['corrected_dbz'].values,
            dbz_values,
            decimal=1
        )

    def test_single_level_clutter_correction(self):
        """Test correction when only the lowest level is contaminated."""
        heights = np.array([100, 300, 500, 700, 900])
        # 100m level has clutter (high value), then normal profile above
        dbz_values = np.array([35.0, 20.0, 22.0, 24.0, 26.0])
        counts = np.array([100, 100, 100, 100, 100])

        ds = xr.Dataset(
            {
                'corrected_dbz': ('height', dbz_values),
                'count': ('height', counts),
            },
            coords={'height': heights},
            attrs={'lowest_level_offset_m': 100, 'freezing_level_m': 2000}
        )

        ds_corrected = remove_ground_clutter(ds)

        # The 100m level should be corrected downward
        # Based on algorithm: corrected = ref_dbz - MKKYNNYS * STEP
        # where ref is at 300m: 20 - (-0.005 * 200) = 20 + 1 = 21
        corrected_100 = ds_corrected['corrected_dbz'].sel(height=100).values

        # Should be reduced from 35 to something close to 21
        assert corrected_100 < 35.0
        assert corrected_100 > 15.0  # Reasonable range

    def test_skip_correction_low_freezing_level(self):
        """Test that correction is skipped when freezing level is 0-1000m."""
        heights = np.array([100, 300, 500, 700])
        dbz_values = np.array([35.0, 20.0, 22.0, 24.0])  # Clear clutter at 100m
        counts = np.array([100, 100, 100, 100])

        ds = xr.Dataset(
            {
                'corrected_dbz': ('height', dbz_values),
                'count': ('height', counts),
            },
            coords={'height': heights},
            attrs={'lowest_level_offset_m': 100, 'freezing_level_m': 500}
        )

        ds_corrected = remove_ground_clutter(ds)

        # Should be unchanged due to low freezing level
        np.testing.assert_array_equal(
            ds_corrected['corrected_dbz'].values,
            dbz_values
        )

    def test_correction_with_high_freezing_level(self):
        """Test that correction proceeds when freezing level > 1000m."""
        heights = np.array([100, 300, 500, 700])
        dbz_values = np.array([35.0, 20.0, 22.0, 24.0])
        counts = np.array([100, 100, 100, 100])

        ds = xr.Dataset(
            {
                'corrected_dbz': ('height', dbz_values),
                'count': ('height', counts),
            },
            coords={'height': heights},
            attrs={'lowest_level_offset_m': 100, 'freezing_level_m': 2000}
        )

        ds_corrected = remove_ground_clutter(ds)

        # Should apply correction (value at 100m should change)
        assert ds_corrected['corrected_dbz'].sel(height=100).values != dbz_values[0]

    def test_correction_with_zero_freezing_level(self):
        """Test that correction proceeds when freezing level = 0."""
        heights = np.array([100, 300, 500, 700])
        dbz_values = np.array([35.0, 20.0, 22.0, 24.0])
        counts = np.array([100, 100, 100, 100])

        ds = xr.Dataset(
            {
                'corrected_dbz': ('height', dbz_values),
                'count': ('height', counts),
            },
            coords={'height': heights},
            attrs={'lowest_level_offset_m': 100, 'freezing_level_m': 0}
        )

        ds_corrected = remove_ground_clutter(ds)

        # Should apply correction even with FL=0
        assert ds_corrected['corrected_dbz'].sel(height=100).values != dbz_values[0]

    def test_mds_levels_set_correctly(self):
        """Test that levels below MDS are handled correctly."""
        heights = np.array([100, 300, 500, 700])
        dbz_values = np.array([MDS, 20.0, 22.0, 24.0])  # Lowest is at MDS (no echo)
        counts = np.array([100, 100, 100, 100])

        ds = xr.Dataset(
            {
                'corrected_dbz': ('height', dbz_values),
                'count': ('height', counts),
            },
            coords={'height': heights},
            attrs={'lowest_level_offset_m': 100, 'freezing_level_m': 2000}
        )

        ds_corrected = remove_ground_clutter(ds)

        # Level at MDS should remain unchanged (no echo to correct)
        assert ds_corrected['corrected_dbz'].sel(height=100).values == MDS

    def test_protection_against_overcorrection(self):
        """Test that correction doesn't increase reflectivity values."""
        heights = np.array([100, 300, 500, 700])
        # Artificial case where extrapolation might increase value
        dbz_values = np.array([18.0, 20.0, 22.0, 24.0])
        counts = np.array([100, 100, 100, 100])

        ds = xr.Dataset(
            {
                'corrected_dbz': ('height', dbz_values),
                'count': ('height', counts),
            },
            coords={'height': heights},
            attrs={'lowest_level_offset_m': 100, 'freezing_level_m': 2000}
        )

        ds_corrected = remove_ground_clutter(ds)

        # All corrected values should be <= original values
        assert (ds_corrected['corrected_dbz'].values <= ds['corrected_dbz'].values + 0.1).all()

    def test_original_dataset_not_modified(self):
        """Test that the input dataset is not modified (immutability)."""
        heights = np.array([100, 300, 500, 700])
        dbz_values = np.array([35.0, 20.0, 22.0, 24.0])
        counts = np.array([100, 100, 100, 100])

        ds = xr.Dataset(
            {
                'corrected_dbz': ('height', dbz_values.copy()),
                'count': ('height', counts),
            },
            coords={'height': heights},
            attrs={'lowest_level_offset_m': 100, 'freezing_level_m': 2000}
        )

        original_values = ds['corrected_dbz'].values.copy()
        ds_corrected = remove_ground_clutter(ds)

        # Original should be unchanged
        np.testing.assert_array_equal(
            ds['corrected_dbz'].values,
            original_values
        )

    def test_multi_level_contamination(self):
        """Test correction when multiple levels are contaminated."""
        heights = np.array([100, 300, 500, 700, 900])
        # First two levels contaminated (high values, steep negative gradients)
        dbz_values = np.array([40.0, 35.0, 20.0, 22.0, 24.0])
        counts = np.array([100, 100, 100, 100, 100])

        ds = xr.Dataset(
            {
                'corrected_dbz': ('height', dbz_values),
                'count': ('height', counts),
            },
            coords={'height': heights},
            attrs={'lowest_level_offset_m': 100, 'freezing_level_m': 2000}
        )

        ds_corrected = remove_ground_clutter(ds)

        # Both 100m and 300m should be corrected downward
        assert ds_corrected['corrected_dbz'].sel(height=100).values < 40.0
        assert ds_corrected['corrected_dbz'].sel(height=300).values < 35.0

        # Higher levels should be relatively unchanged
        assert abs(ds_corrected['corrected_dbz'].sel(height=700).values - 22.0) < 1.0
