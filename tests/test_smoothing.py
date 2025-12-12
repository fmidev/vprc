"""Tests for spike smoothing operations."""

import numpy as np
import xarray as xr
import pytest
from vprc.smoothing import (
    correct_lower_boundary_spike,
    correct_upper_boundary_spike,
    smooth_positive_spikes,
    smooth_negative_spikes,
    fill_gap,
    remove_isolated_echo,
    smooth_spikes,
)
from vprc.constants import MDS


@pytest.fixture
def sample_profile():
    """Create a sample VVP profile for testing."""
    heights = np.arange(100, 3100, 200)  # 100m to 3000m in 200m steps
    times = np.arange(10)  # 10 time points

    # Initialize with missing data
    lin_dbz = np.full((len(heights), len(times)), MDS, dtype=float)
    corrected_dbz = np.full((len(heights), len(times)), MDS, dtype=float)
    sample_count = np.zeros((len(heights), len(times)), dtype=int)

    ds = xr.Dataset(
        {
            'lin_dbz': (['height', 'time'], lin_dbz),
            'corrected_dbz': (['height', 'time'], corrected_dbz),
            'sample_count': (['height', 'time'], sample_count),
        },
        coords={
            'height': heights,
            'time': times,
        }
    )

    return ds


def test_correct_lower_boundary_spike_no_correction_needed(sample_profile):
    """Test that no correction is applied when pattern doesn't match."""
    ds = sample_profile.copy(deep=True)

    # Set up valid data at all levels (no missing data below)
    ds['sample_count'].loc[{'height': 300}] = 50
    ds['sample_count'].loc[{'height': 500}] = 50
    ds['sample_count'].loc[{'height': 700}] = 50
    ds['sample_count'].loc[{'height': 900}] = 50

    ds['lin_dbz'].loc[{'height': 700}] = 20.0
    ds['corrected_dbz'].loc[{'height': 900}] = 15.0

    original = ds['corrected_dbz'].sel(height=700).copy()
    ds_corrected = correct_lower_boundary_spike(ds, height=500)

    # Should not change anything since pattern doesn't match
    np.testing.assert_array_equal(
        ds_corrected['corrected_dbz'].sel(height=700),
        original
    )


def test_correct_lower_boundary_spike_with_positive_gradient(sample_profile):
    """Test correction when gradient above is positive."""
    ds = sample_profile.copy(deep=True)

    # Set up the pattern for height=500:
    # - height=300: missing (sample_count < 30)
    # - height=500, 700, 900: valid data
    ds['sample_count'].loc[{'height': 300}] = 10  # Below threshold
    ds['sample_count'].loc[{'height': 500}] = 50
    ds['sample_count'].loc[{'height': 700}] = 50
    ds['sample_count'].loc[{'height': 900}] = 50

    # Create unphysical increase at boundary
    # lin_dbz at 700m > corrected_dbz at 900m
    ds['lin_dbz'].loc[{'height': 700}] = 25.0
    ds['corrected_dbz'].loc[{'height': 700}] = 25.0
    ds['corrected_dbz'].loc[{'height': 900}] = 20.0
    ds['corrected_dbz'].loc[{'height': 1100}] = 22.0  # Positive gradient

    ds_corrected = correct_lower_boundary_spike(ds, height=500)

    # With positive gradient above (+2 dBZ from 900m to 1100m),
    # should extrapolate: corrected[700] = 20.0 - (22.0 - 20.0) = 18.0
    expected = 18.0
    result = ds_corrected['corrected_dbz'].sel(height=700).values

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_correct_lower_boundary_spike_with_negative_gradient(sample_profile):
    """Test correction when gradient above is negative or zero."""
    ds = sample_profile.copy(deep=True)

    # Set up the pattern
    ds['sample_count'].loc[{'height': 300}] = 10  # Below threshold
    ds['sample_count'].loc[{'height': 500}] = 50
    ds['sample_count'].loc[{'height': 700}] = 50
    ds['sample_count'].loc[{'height': 900}] = 50

    # Create unphysical increase
    ds['lin_dbz'].loc[{'height': 700}] = 25.0
    ds['corrected_dbz'].loc[{'height': 700}] = 25.0
    ds['corrected_dbz'].loc[{'height': 900}] = 20.0
    ds['corrected_dbz'].loc[{'height': 1100}] = 18.0  # Negative gradient

    ds_corrected = correct_lower_boundary_spike(ds, height=500)

    # With negative gradient, should set equal to level above
    # corrected[700] = corrected[900] = 20.0
    expected = 20.0
    result = ds_corrected['corrected_dbz'].sel(height=700).values

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_correct_lower_boundary_spike_multiple_times(sample_profile):
    """Test that correction is applied independently at each time point."""
    ds = sample_profile.copy(deep=True)

    # Set up pattern for all times
    ds['sample_count'].loc[{'height': 300}] = 10
    ds['sample_count'].loc[{'height': 500}] = 50
    ds['sample_count'].loc[{'height': 700}] = 50
    ds['sample_count'].loc[{'height': 900}] = 50

    # Set up unphysical increase for first 5 times only
    ds['lin_dbz'].loc[{'height': 700, 'time': slice(0, 5)}] = 25.0
    ds['corrected_dbz'].loc[{'height': 700, 'time': slice(0, 5)}] = 25.0
    ds['corrected_dbz'].loc[{'height': 900}] = 20.0
    ds['corrected_dbz'].loc[{'height': 1100}] = 22.0

    # For last 5 times, no unphysical increase
    ds['lin_dbz'].loc[{'height': 700, 'time': slice(5, 10)}] = 15.0
    ds['corrected_dbz'].loc[{'height': 700, 'time': slice(5, 10)}] = 15.0

    ds_corrected = correct_lower_boundary_spike(ds, height=500)

    # First 5 times should be corrected to 18.0
    result_corrected = ds_corrected['corrected_dbz'].sel(height=700, time=slice(0, 5)).values
    np.testing.assert_allclose(result_corrected[:5], 18.0, rtol=1e-5)

    # Last 5 times should remain unchanged at 15.0
    result_unchanged = ds_corrected['corrected_dbz'].sel(height=700, time=slice(5, 10)).values
    np.testing.assert_allclose(result_unchanged, 15.0, rtol=1e-5)


def test_correct_lower_boundary_spike_missing_heights():
    """Test that function handles gracefully when required heights don't exist."""
    # Create dataset with limited height range
    heights = np.arange(100, 700, 200)  # Only up to 500m
    times = np.arange(5)

    ds = xr.Dataset(
        {
            'lin_dbz': (['height', 'time'], np.full((len(heights), len(times)), MDS)),
            'corrected_dbz': (['height', 'time'], np.full((len(heights), len(times)), MDS)),
            'sample_count': (['height', 'time'], np.zeros((len(heights), len(times)), dtype=int)),
        },
        coords={'height': heights, 'time': times}
    )

    # Try to process height=500, but 900m and 1100m don't exist
    ds_result = correct_lower_boundary_spike(ds, height=500)

    # Should return unchanged
    xr.testing.assert_identical(ds_result, ds)


def test_correct_upper_boundary_spike_no_correction_needed(sample_profile):
    """Test that no correction is applied when pattern doesn't match."""
    ds = sample_profile.copy(deep=True)

    # Set up valid data at all levels (no missing data above)
    ds['sample_count'].loc[{'height': 500}] = 50
    ds['sample_count'].loc[{'height': 700}] = 50
    ds['sample_count'].loc[{'height': 900}] = 50
    ds['sample_count'].loc[{'height': 1100}] = 50

    ds['lin_dbz'].loc[{'height': 700}] = 20.0
    ds['corrected_dbz'].loc[{'height': 500}] = 15.0

    original = ds['corrected_dbz'].sel(height=700).copy()
    ds_corrected = correct_upper_boundary_spike(ds, height=500)

    # Should not change anything since pattern doesn't match
    np.testing.assert_array_equal(
        ds_corrected['corrected_dbz'].sel(height=700),
        original
    )


def test_correct_upper_boundary_spike_with_negative_gradient(sample_profile):
    """Test correction when gradient below is negative."""
    ds = sample_profile.copy(deep=True)

    # Set up the pattern for height=500:
    # - height=500, 700: valid data
    # - height=900, 1100: missing (sample_count < 30)
    ds['sample_count'].loc[{'height': 500}] = 50
    ds['sample_count'].loc[{'height': 700}] = 50
    ds['sample_count'].loc[{'height': 900}] = 10  # Below threshold
    ds['sample_count'].loc[{'height': 1100}] = 10  # Below threshold

    # Create unphysical increase at boundary
    # lin_dbz at 700m > corrected_dbz at 500m
    ds['lin_dbz'].loc[{'height': 700}] = 25.0
    ds['corrected_dbz'].loc[{'height': 700}] = 25.0
    ds['corrected_dbz'].loc[{'height': 500}] = 20.0
    ds['corrected_dbz'].loc[{'height': 300}] = 22.0  # Negative gradient (22 -> 20)

    ds_corrected = correct_upper_boundary_spike(ds, height=500)

    # With negative gradient below (-2 dBZ from 300m to 500m),
    # should extrapolate: corrected[700] = 20.0 + (20.0 - 22.0) = 18.0
    expected = 18.0
    result = ds_corrected['corrected_dbz'].sel(height=700).values

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_correct_upper_boundary_spike_with_positive_gradient(sample_profile):
    """Test correction when gradient below is positive or zero."""
    ds = sample_profile.copy(deep=True)

    # Set up the pattern
    ds['sample_count'].loc[{'height': 500}] = 50
    ds['sample_count'].loc[{'height': 700}] = 50
    ds['sample_count'].loc[{'height': 900}] = 10  # Below threshold
    ds['sample_count'].loc[{'height': 1100}] = 10  # Below threshold

    # Create unphysical increase
    ds['lin_dbz'].loc[{'height': 700}] = 25.0
    ds['corrected_dbz'].loc[{'height': 700}] = 25.0
    ds['corrected_dbz'].loc[{'height': 500}] = 20.0
    ds['corrected_dbz'].loc[{'height': 300}] = 18.0  # Positive gradient (18 -> 20)

    ds_corrected = correct_upper_boundary_spike(ds, height=500)

    # With positive gradient, should set equal to level below
    # corrected[700] = corrected[500] = 20.0
    expected = 20.0
    result = ds_corrected['corrected_dbz'].sel(height=700).values

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_correct_upper_boundary_spike_multiple_times(sample_profile):
    """Test that correction is applied independently at each time point."""
    ds = sample_profile.copy(deep=True)

    # Set up pattern for all times
    ds['sample_count'].loc[{'height': 500}] = 50
    ds['sample_count'].loc[{'height': 700}] = 50
    ds['sample_count'].loc[{'height': 900}] = 10
    ds['sample_count'].loc[{'height': 1100}] = 10

    # Set up unphysical increase for first 5 times only
    ds['lin_dbz'].loc[{'height': 700, 'time': slice(0, 5)}] = 25.0
    ds['corrected_dbz'].loc[{'height': 700, 'time': slice(0, 5)}] = 25.0
    ds['corrected_dbz'].loc[{'height': 500}] = 20.0
    ds['corrected_dbz'].loc[{'height': 300}] = 22.0  # Negative gradient

    # For last 5 times, no unphysical increase
    ds['lin_dbz'].loc[{'height': 700, 'time': slice(5, 10)}] = 15.0
    ds['corrected_dbz'].loc[{'height': 700, 'time': slice(5, 10)}] = 15.0

    ds_corrected = correct_upper_boundary_spike(ds, height=500)

    # First 5 times should be corrected to 18.0
    result_corrected = ds_corrected['corrected_dbz'].sel(height=700, time=slice(0, 5)).values
    np.testing.assert_allclose(result_corrected[:5], 18.0, rtol=1e-5)

    # Last 5 times should remain unchanged at 15.0
    result_unchanged = ds_corrected['corrected_dbz'].sel(height=700, time=slice(5, 10)).values
    np.testing.assert_allclose(result_unchanged, 15.0, rtol=1e-5)


def test_correct_upper_boundary_spike_missing_heights():
    """Test that function handles gracefully when required heights don't exist."""
    # Create dataset with limited height range
    heights = np.arange(100, 700, 200)  # Only up to 500m
    times = np.arange(5)

    ds = xr.Dataset(
        {
            'lin_dbz': (['height', 'time'], np.full((len(heights), len(times)), MDS)),
            'corrected_dbz': (['height', 'time'], np.full((len(heights), len(times)), MDS)),
            'sample_count': (['height', 'time'], np.zeros((len(heights), len(times)), dtype=int)),
        },
        coords={'height': heights, 'time': times}
    )

    # Try to process height=500, but 900m and 1100m don't exist
    ds_result = correct_upper_boundary_spike(ds, height=500)

    # Should return unchanged
    xr.testing.assert_identical(ds_result, ds)


def test_both_boundary_corrections_together(sample_profile):
    """Test that both lower and upper boundary corrections work together."""
    ds = sample_profile.copy(deep=True)

    # Set up lower boundary spike at height=500
    ds['sample_count'].loc[{'height': 300}] = 10  # Missing
    ds['sample_count'].loc[{'height': 500}] = 50
    ds['sample_count'].loc[{'height': 700}] = 50
    ds['sample_count'].loc[{'height': 900}] = 50
    ds['lin_dbz'].loc[{'height': 700}] = 25.0
    ds['corrected_dbz'].loc[{'height': 700}] = 25.0
    ds['corrected_dbz'].loc[{'height': 900}] = 20.0
    ds['corrected_dbz'].loc[{'height': 1100}] = 22.0

    # Set up upper boundary spike at height=1500
    ds['sample_count'].loc[{'height': 1500}] = 50
    ds['sample_count'].loc[{'height': 1700}] = 50
    ds['sample_count'].loc[{'height': 1900}] = 10  # Missing
    ds['sample_count'].loc[{'height': 2100}] = 10  # Missing
    ds['lin_dbz'].loc[{'height': 1700}] = 30.0
    ds['corrected_dbz'].loc[{'height': 1700}] = 30.0
    ds['corrected_dbz'].loc[{'height': 1500}] = 25.0
    ds['corrected_dbz'].loc[{'height': 1300}] = 27.0  # Negative gradient

    # Apply both corrections
    ds_corrected = ds.copy(deep=True)
    for height in sorted(ds.height.values):
        ds_corrected = correct_lower_boundary_spike(ds_corrected, height)
        ds_corrected = correct_upper_boundary_spike(ds_corrected, height)

    # Lower boundary should be corrected to 18.0
    np.testing.assert_allclose(
        ds_corrected['corrected_dbz'].sel(height=700).values,
        18.0,
        rtol=1e-5
    )

    # Upper boundary should be corrected to 23.0 (25 + (25-27))
    np.testing.assert_allclose(
        ds_corrected['corrected_dbz'].sel(height=1700).values,
        23.0,
        rtol=1e-5
    )


def test_smooth_positive_spikes_no_spikes(sample_profile):
    """Test that no correction is applied when there are no spikes."""
    ds = sample_profile.copy(deep=True)

    # Set up smooth profile with no spikes
    ds['sample_count'].loc[{'height': slice(500, 1100)}] = 50
    ds['lin_dbz'].loc[{'height': 500}] = 18.0
    ds['lin_dbz'].loc[{'height': 700}] = 20.0
    ds['lin_dbz'].loc[{'height': 900}] = 22.0
    ds['corrected_dbz'].loc[{'height': 500}] = 18.0
    ds['corrected_dbz'].loc[{'height': 700}] = 20.0
    ds['corrected_dbz'].loc[{'height': 900}] = 22.0

    original = ds['corrected_dbz'].copy()
    ds_smoothed = smooth_positive_spikes(ds)

    # Should not change anything (differences < 3 dBZ)
    xr.testing.assert_identical(ds_smoothed['corrected_dbz'], original)


def test_smooth_positive_spikes_small_spike_3point_average(sample_profile):
    """Test that small positive spikes are smoothed with 3-point average."""
    ds = sample_profile.copy(deep=True)

    # Set up a moderate positive spike at 700m
    # Use a spike where 3-point average will be sufficient
    ds['sample_count'].loc[{'height': slice(500, 1100)}] = 50
    ds['lin_dbz'].loc[{'height': 500}] = 16.0
    ds['lin_dbz'].loc[{'height': 700}] = 20.0  # Spike: 4 dBZ above both neighbors
    ds['lin_dbz'].loc[{'height': 900}] = 16.0
    ds['corrected_dbz'].loc[{'height': 500}] = 16.0
    ds['corrected_dbz'].loc[{'height': 700}] = 20.0
    ds['corrected_dbz'].loc[{'height': 900}] = 16.0

    ds_smoothed = smooth_positive_spikes(ds)

    # 3-point average: (16 + 20 + 16) / 3 = 17.333...
    # Difference: 20 - 17.333 = 2.667, which is NOT > 4 dBZ threshold
    # So should use 3-point average
    expected = (16.0 + 20.0 + 16.0) / 3
    result = ds_smoothed['corrected_dbz'].sel(height=700).values

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_smooth_positive_spikes_large_spike_2point_average(sample_profile):
    """Test that large positive spikes are smoothed with 2-point average."""
    ds = sample_profile.copy(deep=True)

    # Set up a large positive spike at 700m
    ds['sample_count'].loc[{'height': slice(500, 1100)}] = 50
    ds['lin_dbz'].loc[{'height': 500}] = 15.0
    ds['lin_dbz'].loc[{'height': 700}] = 30.0  # Large spike: 15 dBZ above neighbors
    ds['lin_dbz'].loc[{'height': 900}] = 15.0
    ds['corrected_dbz'].loc[{'height': 500}] = 15.0
    ds['corrected_dbz'].loc[{'height': 700}] = 30.0
    ds['corrected_dbz'].loc[{'height': 900}] = 15.0

    ds_smoothed = smooth_positive_spikes(ds)

    # 3-point average would be (15 + 30 + 15) / 3 = 20
    # Difference: 30 - 20 = 10 dBZ, which is > 4 dBZ threshold
    # So should use 2-point average: (15 + 15) / 2 = 15
    expected = 15.0
    result = ds_smoothed['corrected_dbz'].sel(height=700).values

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_smooth_positive_spikes_below_threshold(sample_profile):
    """Test that spikes below 3 dBZ threshold are not smoothed."""
    ds = sample_profile.copy(deep=True)

    # Small elevation that doesn't qualify as spike
    ds['sample_count'].loc[{'height': slice(500, 1100)}] = 50
    ds['lin_dbz'].loc[{'height': 500}] = 15.0
    ds['lin_dbz'].loc[{'height': 700}] = 17.9  # 2.9 dBZ above (below threshold)
    ds['lin_dbz'].loc[{'height': 900}] = 15.0
    ds['corrected_dbz'].loc[{'height': 500}] = 15.0
    ds['corrected_dbz'].loc[{'height': 700}] = 17.9
    ds['corrected_dbz'].loc[{'height': 900}] = 15.0

    original = ds['corrected_dbz'].copy()
    ds_smoothed = smooth_positive_spikes(ds)

    # Should not be modified
    xr.testing.assert_identical(ds_smoothed['corrected_dbz'], original)


def test_smooth_positive_spikes_multiple_spikes(sample_profile):
    """Test smoothing multiple spikes at different heights."""
    ds = sample_profile.copy(deep=True)

    # Set up multiple spikes
    ds['sample_count'].loc[{'height': slice(500, 1700)}] = 50

    # Spike 1 at 700m (moderate - will use 3-point)
    ds['lin_dbz'].loc[{'height': 500}] = 16.0
    ds['lin_dbz'].loc[{'height': 700}] = 20.0  # 4 dBZ spike
    ds['lin_dbz'].loc[{'height': 900}] = 16.0
    ds['corrected_dbz'].loc[{'height': 500}] = 16.0
    ds['corrected_dbz'].loc[{'height': 700}] = 20.0
    ds['corrected_dbz'].loc[{'height': 900}] = 16.0

    # Spike 2 at 1300m (large - will use 2-point)
    ds['lin_dbz'].loc[{'height': 1100}] = 18.0
    ds['lin_dbz'].loc[{'height': 1300}] = 35.0  # 17 dBZ spike
    ds['lin_dbz'].loc[{'height': 1500}] = 18.0
    ds['corrected_dbz'].loc[{'height': 1100}] = 18.0
    ds['corrected_dbz'].loc[{'height': 1300}] = 35.0
    ds['corrected_dbz'].loc[{'height': 1500}] = 18.0

    ds_smoothed = smooth_positive_spikes(ds)

    # Spike 1: 3-point avg = (16+20+16)/3 = 17.333, diff = 20-17.333 = 2.667 < 4
    # Should use 3-point average
    expected_1 = (16.0 + 20.0 + 16.0) / 3
    result_1 = ds_smoothed['corrected_dbz'].sel(height=700).values
    np.testing.assert_allclose(result_1, expected_1, rtol=1e-5)

    # Spike 2: 3-point avg = (18+35+18)/3 = 23.667, diff = 35-23.667 = 11.333 > 4
    # Should use 2-point average
    expected_2 = (18.0 + 18.0) / 2
    result_2 = ds_smoothed['corrected_dbz'].sel(height=1300).values
    np.testing.assert_allclose(result_2, expected_2, rtol=1e-5)


def test_smooth_positive_spikes_invalid_data_ignored(sample_profile):
    """Test that spikes in invalid data (low sample count) are ignored."""
    ds = sample_profile.copy(deep=True)

    # Set up spike but with low sample count at middle level
    ds['sample_count'].loc[{'height': 500}] = 50
    ds['sample_count'].loc[{'height': 700}] = 10  # Below threshold!
    ds['sample_count'].loc[{'height': 900}] = 50
    ds['lin_dbz'].loc[{'height': 500}] = 15.0
    ds['lin_dbz'].loc[{'height': 700}] = 25.0  # Would be a spike
    ds['lin_dbz'].loc[{'height': 900}] = 15.0
    ds['corrected_dbz'].loc[{'height': 500}] = 15.0
    ds['corrected_dbz'].loc[{'height': 700}] = 25.0
    ds['corrected_dbz'].loc[{'height': 900}] = 15.0

    original = ds['corrected_dbz'].copy()
    ds_smoothed = smooth_positive_spikes(ds)

    # Should not be modified because middle point has insufficient samples
    xr.testing.assert_identical(ds_smoothed['corrected_dbz'], original)


def test_smooth_positive_spikes_asymmetric_spike(sample_profile):
    """Test that asymmetric peaks are not smoothed (must exceed BOTH neighbors)."""
    ds = sample_profile.copy(deep=True)

    # Spike that only exceeds one neighbor
    ds['sample_count'].loc[{'height': slice(500, 1100)}] = 50
    ds['lin_dbz'].loc[{'height': 500}] = 15.0
    ds['lin_dbz'].loc[{'height': 700}] = 22.0  # 7 above lower, but only 2 above upper
    ds['lin_dbz'].loc[{'height': 900}] = 20.0
    ds['corrected_dbz'].loc[{'height': 500}] = 15.0
    ds['corrected_dbz'].loc[{'height': 700}] = 22.0
    ds['corrected_dbz'].loc[{'height': 900}] = 20.0

    original = ds['corrected_dbz'].copy()
    ds_smoothed = smooth_positive_spikes(ds)

    # Should not be modified (doesn't exceed upper neighbor by >3 dBZ)
    xr.testing.assert_identical(ds_smoothed['corrected_dbz'], original)


def test_smooth_negative_spikes_no_spikes(sample_profile):
    """Test that no correction is applied when there are no negative spikes."""
    ds = sample_profile.copy(deep=True)

    # Set up smooth profile with no spikes
    ds['sample_count'].loc[{'height': slice(500, 1100)}] = 50
    ds['corrected_dbz'].loc[{'height': 500}] = 18.0
    ds['corrected_dbz'].loc[{'height': 700}] = 20.0
    ds['corrected_dbz'].loc[{'height': 900}] = 22.0

    original = ds['corrected_dbz'].copy()
    ds_smoothed = smooth_negative_spikes(ds)

    # Should not change anything
    xr.testing.assert_identical(ds_smoothed['corrected_dbz'], original)


def test_smooth_negative_spikes_small_spike_3point_average(sample_profile):
    """Test that small negative spikes are smoothed with 3-point average."""
    ds = sample_profile.copy(deep=True)

    # Set up a moderate negative spike at 700m
    ds['sample_count'].loc[{'height': slice(500, 1100)}] = 50
    ds['corrected_dbz'].loc[{'height': 500}] = 20.0
    ds['corrected_dbz'].loc[{'height': 700}] = 16.0  # 4 dBZ below neighbors
    ds['corrected_dbz'].loc[{'height': 900}] = 20.0

    ds_smoothed = smooth_negative_spikes(ds)

    # 3-point average: (20 + 16 + 20) / 3 = 18.667
    # Difference: 16 - 18.667 = -2.667, which is NOT < -10 dBZ threshold
    # So should use 3-point average
    expected = (20.0 + 16.0 + 20.0) / 3
    result = ds_smoothed['corrected_dbz'].sel(height=700).values

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_smooth_negative_spikes_large_spike_2point_average(sample_profile):
    """Test that large negative spikes are smoothed with 2-point average."""
    ds = sample_profile.copy(deep=True)

    # Set up a large negative spike at 700m
    ds['sample_count'].loc[{'height': slice(500, 1100)}] = 50
    ds['corrected_dbz'].loc[{'height': 500}] = 25.0
    ds['corrected_dbz'].loc[{'height': 700}] = 5.0   # 20 dBZ below neighbors
    ds['corrected_dbz'].loc[{'height': 900}] = 25.0

    ds_smoothed = smooth_negative_spikes(ds)

    # 3-point average would be (25 + 5 + 25) / 3 = 18.333
    # Difference: 5 - 18.333 = -13.333 dBZ, which is < -10 dBZ threshold
    # So should use 2-point average: (25 + 25) / 2 = 25
    expected = 25.0
    result = ds_smoothed['corrected_dbz'].sel(height=700).values

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_smooth_negative_spikes_exactly_at_threshold(sample_profile):
    """Test negative spike exactly at the 3 dBZ detection threshold."""
    ds = sample_profile.copy(deep=True)

    # Spike that just barely qualifies (<-3 dBZ below neighbors)
    ds['sample_count'].loc[{'height': slice(500, 1100)}] = 50
    ds['corrected_dbz'].loc[{'height': 500}] = 20.0
    ds['corrected_dbz'].loc[{'height': 700}] = 16.9  # 3.1 dBZ below
    ds['corrected_dbz'].loc[{'height': 900}] = 20.0

    ds_smoothed = smooth_negative_spikes(ds)

    # Should be smoothed with 3-point average
    expected = (20.0 + 16.9 + 20.0) / 3
    result = ds_smoothed['corrected_dbz'].sel(height=700).values

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_smooth_negative_spikes_below_threshold(sample_profile):
    """Test that negative dips below 3 dBZ threshold are not smoothed."""
    ds = sample_profile.copy(deep=True)

    # Small dip that doesn't qualify as spike
    ds['sample_count'].loc[{'height': slice(500, 1100)}] = 50
    ds['corrected_dbz'].loc[{'height': 500}] = 20.0
    ds['corrected_dbz'].loc[{'height': 700}] = 17.1  # 2.9 dBZ below (above threshold)
    ds['corrected_dbz'].loc[{'height': 900}] = 20.0

    original = ds['corrected_dbz'].copy()
    ds_smoothed = smooth_negative_spikes(ds)

    # Should not be modified
    xr.testing.assert_identical(ds_smoothed['corrected_dbz'], original)


def test_smooth_negative_spikes_multiple_spikes(sample_profile):
    """Test smoothing multiple negative spikes at different heights."""
    ds = sample_profile.copy(deep=True)

    # Set up multiple spikes
    ds['sample_count'].loc[{'height': slice(500, 1700)}] = 50

    # Spike 1 at 700m (moderate - will use 3-point)
    ds['corrected_dbz'].loc[{'height': 500}] = 20.0
    ds['corrected_dbz'].loc[{'height': 700}] = 16.0  # 4 dBZ dip
    ds['corrected_dbz'].loc[{'height': 900}] = 20.0

    # Spike 2 at 1300m (large - will use 2-point)
    ds['corrected_dbz'].loc[{'height': 1100}] = 25.0
    ds['corrected_dbz'].loc[{'height': 1300}] = 5.0   # 20 dBZ dip
    ds['corrected_dbz'].loc[{'height': 1500}] = 25.0

    ds_smoothed = smooth_negative_spikes(ds)

    # Spike 1: 3-point avg = (20+16+20)/3 = 18.667, diff = 16-18.667 = -2.667 > -10
    # Should use 3-point average
    expected_1 = (20.0 + 16.0 + 20.0) / 3
    result_1 = ds_smoothed['corrected_dbz'].sel(height=700).values
    np.testing.assert_allclose(result_1, expected_1, rtol=1e-5)

    # Spike 2: 3-point avg = (25+5+25)/3 = 18.333, diff = 5-18.333 = -13.333 < -10
    # Should use 2-point average
    expected_2 = (25.0 + 25.0) / 2
    result_2 = ds_smoothed['corrected_dbz'].sel(height=1300).values
    np.testing.assert_allclose(result_2, expected_2, rtol=1e-5)


def test_smooth_negative_spikes_invalid_data_ignored(sample_profile):
    """Test that negative spikes in invalid data (low sample count) are ignored."""
    ds = sample_profile.copy(deep=True)

    # Set up spike but with low sample count at middle level
    ds['sample_count'].loc[{'height': 500}] = 50
    ds['sample_count'].loc[{'height': 700}] = 10  # Below threshold!
    ds['sample_count'].loc[{'height': 900}] = 50
    ds['corrected_dbz'].loc[{'height': 500}] = 25.0
    ds['corrected_dbz'].loc[{'height': 700}] = 10.0  # Would be a spike
    ds['corrected_dbz'].loc[{'height': 900}] = 25.0

    original = ds['corrected_dbz'].copy()
    ds_smoothed = smooth_negative_spikes(ds)

    # Should not be modified because middle point has insufficient samples
    xr.testing.assert_identical(ds_smoothed['corrected_dbz'], original)


def test_smooth_negative_spikes_asymmetric_dip(sample_profile):
    """Test that asymmetric dips are not smoothed (must be below BOTH neighbors)."""
    ds = sample_profile.copy(deep=True)

    # Dip that is only below one neighbor
    ds['sample_count'].loc[{'height': slice(500, 1100)}] = 50
    ds['corrected_dbz'].loc[{'height': 500}] = 25.0
    ds['corrected_dbz'].loc[{'height': 700}] = 18.0  # 7 below lower, but only 2 below upper
    ds['corrected_dbz'].loc[{'height': 900}] = 20.0

    original = ds['corrected_dbz'].copy()
    ds_smoothed = smooth_negative_spikes(ds)

    # Should not be modified (doesn't exceed lower neighbor by <-3 dBZ)
    xr.testing.assert_identical(ds_smoothed['corrected_dbz'], original)


def test_both_positive_and_negative_spikes_together(sample_profile):
    """Test that both positive and negative spike smoothing work together."""
    ds = sample_profile.copy(deep=True)

    ds['sample_count'].loc[{'height': slice(500, 2100)}] = 50

    # Positive spike at 700m
    ds['corrected_dbz'].loc[{'height': 500}] = 16.0
    ds['corrected_dbz'].loc[{'height': 700}] = 20.0  # 4 dBZ spike
    ds['corrected_dbz'].loc[{'height': 900}] = 16.0

    # Smooth transition region (no spikes)
    ds['corrected_dbz'].loc[{'height': 1100}] = 18.0
    ds['corrected_dbz'].loc[{'height': 1300}] = 20.0
    ds['corrected_dbz'].loc[{'height': 1500}] = 22.0

    # Negative spike at 1900m (well separated)
    ds['corrected_dbz'].loc[{'height': 1700}] = 24.0
    ds['corrected_dbz'].loc[{'height': 1900}] = 20.0  # 4 dBZ dip
    ds['corrected_dbz'].loc[{'height': 2100}] = 24.0

    # Apply both smoothing operations
    ds_smoothed = smooth_positive_spikes(ds)
    ds_smoothed = smooth_negative_spikes(ds_smoothed)

    # Positive spike: 3-point avg = (16+20+16)/3 = 17.333...
    expected_pos = (16.0 + 20.0 + 16.0) / 3
    result_pos = ds_smoothed['corrected_dbz'].sel(height=700).values
    np.testing.assert_allclose(result_pos, expected_pos, rtol=1e-5)

    # Negative spike: 3-point avg = (24+20+24)/3 = 22.666...
    expected_neg = (24.0 + 20.0 + 24.0) / 3
    result_neg = ds_smoothed['corrected_dbz'].sel(height=1900).values
    np.testing.assert_allclose(result_neg, expected_neg, rtol=1e-5)


# =============================================================================
# Tests for fill_gap (Operation 5)
# =============================================================================


def test_fill_gap_no_gap(sample_profile):
    """Test that no fill is applied when there is no gap."""
    ds = sample_profile.copy(deep=True)

    # All valid data - no gaps
    ds['sample_count'].loc[{'height': slice(500, 1300)}] = 50
    ds['corrected_dbz'].loc[{'height': 500}] = 10.0
    ds['corrected_dbz'].loc[{'height': 700}] = 15.0
    ds['corrected_dbz'].loc[{'height': 900}] = 20.0
    ds['corrected_dbz'].loc[{'height': 1100}] = 18.0
    ds['corrected_dbz'].loc[{'height': 1300}] = 16.0

    original = ds['corrected_dbz'].copy()
    ds_filled = fill_gap(ds, 900)

    xr.testing.assert_identical(ds_filled['corrected_dbz'], original)


def test_fill_gap_pattern_a(sample_profile):
    """Test gap filling with 2 valid levels below + 1 valid level above."""
    ds = sample_profile.copy(deep=True)

    # Pattern A: 2 valid below, 1 valid above
    ds['sample_count'].loc[{'height': 500}] = 50  # valid (-400)
    ds['sample_count'].loc[{'height': 700}] = 50  # valid (-200)
    ds['sample_count'].loc[{'height': 900}] = 10  # GAP (< threshold)
    ds['sample_count'].loc[{'height': 1100}] = 50  # valid (+200)
    ds['sample_count'].loc[{'height': 1300}] = 10  # invalid (+400) - only 1 above

    ds['corrected_dbz'].loc[{'height': 500}] = 10.0
    ds['corrected_dbz'].loc[{'height': 700}] = 15.0
    ds['corrected_dbz'].loc[{'height': 900}] = 0.0  # will be filled
    ds['corrected_dbz'].loc[{'height': 1100}] = 25.0
    ds['corrected_dbz'].loc[{'height': 1300}] = MDS

    ds_filled = fill_gap(ds, 900)

    # Should be average of adjacent: (15 + 25) / 2 = 20
    expected = (15.0 + 25.0) / 2
    result = ds_filled['corrected_dbz'].sel(height=900).values
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_fill_gap_pattern_b(sample_profile):
    """Test gap filling with 1 valid level below + 2 valid levels above."""
    ds = sample_profile.copy(deep=True)

    # Pattern B: 1 valid below, 2 valid above
    ds['sample_count'].loc[{'height': 500}] = 10  # invalid (-400)
    ds['sample_count'].loc[{'height': 700}] = 50  # valid (-200)
    ds['sample_count'].loc[{'height': 900}] = 10  # GAP (< threshold)
    ds['sample_count'].loc[{'height': 1100}] = 50  # valid (+200)
    ds['sample_count'].loc[{'height': 1300}] = 50  # valid (+400)

    ds['corrected_dbz'].loc[{'height': 500}] = MDS
    ds['corrected_dbz'].loc[{'height': 700}] = 12.0
    ds['corrected_dbz'].loc[{'height': 900}] = 0.0  # will be filled
    ds['corrected_dbz'].loc[{'height': 1100}] = 18.0
    ds['corrected_dbz'].loc[{'height': 1300}] = 20.0

    ds_filled = fill_gap(ds, 900)

    # Should be average of adjacent: (12 + 18) / 2 = 15
    expected = (12.0 + 18.0) / 2
    result = ds_filled['corrected_dbz'].sel(height=900).values
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_fill_gap_not_filled_insufficient_context(sample_profile):
    """Test that gaps are not filled without sufficient context."""
    ds = sample_profile.copy(deep=True)

    # Only 1 valid below and 1 valid above - not enough
    ds['sample_count'].loc[{'height': 500}] = 10  # invalid
    ds['sample_count'].loc[{'height': 700}] = 50  # valid
    ds['sample_count'].loc[{'height': 900}] = 10  # GAP
    ds['sample_count'].loc[{'height': 1100}] = 50  # valid
    ds['sample_count'].loc[{'height': 1300}] = 10  # invalid

    ds['corrected_dbz'].loc[{'height': 700}] = 15.0
    ds['corrected_dbz'].loc[{'height': 900}] = 0.0
    ds['corrected_dbz'].loc[{'height': 1100}] = 25.0

    original = ds['corrected_dbz'].sel(height=900).copy()
    ds_filled = fill_gap(ds, 900)

    # Should NOT be filled
    xr.testing.assert_identical(ds_filled['corrected_dbz'].sel(height=900), original)


def test_fill_gap_requires_corrected_above_mds(sample_profile):
    """Test that gap is not filled if corrected values are at MDS."""
    ds = sample_profile.copy(deep=True)

    # Pattern A setup, but adjacent corrected value is MDS
    ds['sample_count'].loc[{'height': 500}] = 50
    ds['sample_count'].loc[{'height': 700}] = 50
    ds['sample_count'].loc[{'height': 900}] = 10  # GAP
    ds['sample_count'].loc[{'height': 1100}] = 50

    ds['corrected_dbz'].loc[{'height': 500}] = 10.0
    ds['corrected_dbz'].loc[{'height': 700}] = 15.0
    ds['corrected_dbz'].loc[{'height': 900}] = 0.0
    ds['corrected_dbz'].loc[{'height': 1100}] = MDS  # At MDS - should prevent fill

    original = ds['corrected_dbz'].sel(height=900).copy()
    ds_filled = fill_gap(ds, 900)

    # Should NOT be filled because adjacent is at MDS
    xr.testing.assert_identical(ds_filled['corrected_dbz'].sel(height=900), original)


def test_fill_gap_at_edge(sample_profile):
    """Test that fill_gap handles edge heights gracefully."""
    ds = sample_profile.copy(deep=True)

    # Try to fill at lowest height - should return unchanged
    ds_filled = fill_gap(ds, 100)

    xr.testing.assert_identical(ds_filled, ds)


# =============================================================================
# Tests for remove_isolated_echo (Operation 6)
# =============================================================================


def test_remove_isolated_echo_no_isolated(sample_profile):
    """Test that continuous echoes are not removed."""
    ds = sample_profile.copy(deep=True)

    # All valid data - no isolated echoes
    ds['sample_count'].loc[{'height': slice(500, 1300)}] = 50
    ds['corrected_dbz'].loc[{'height': slice(500, 1300)}] = 20.0

    original = ds['corrected_dbz'].copy()
    ds_cleaned = remove_isolated_echo(ds, 900)

    xr.testing.assert_identical(ds_cleaned['corrected_dbz'], original)


def test_remove_isolated_echo_detected(sample_profile):
    """Test that isolated echoes are removed."""
    ds = sample_profile.copy(deep=True)

    # Isolated echo at 900m - all neighbors invalid
    ds['sample_count'].loc[{'height': 500}] = 10  # invalid (-400)
    ds['sample_count'].loc[{'height': 700}] = 10  # invalid (-200)
    ds['sample_count'].loc[{'height': 900}] = 50  # VALID - isolated
    ds['sample_count'].loc[{'height': 1100}] = 10  # invalid (+200)
    ds['sample_count'].loc[{'height': 1300}] = 10  # invalid (+400)

    ds['corrected_dbz'].loc[{'height': 900}] = 25.0  # will be removed

    ds_cleaned = remove_isolated_echo(ds, 900)

    # Should be set to MDS
    result = ds_cleaned['corrected_dbz'].sel(height=900).values
    np.testing.assert_array_equal(result, MDS)


def test_remove_isolated_echo_one_neighbor_valid(sample_profile):
    """Test that echo with one valid neighbor is not removed."""
    ds = sample_profile.copy(deep=True)

    # Echo at 900m with one valid neighbor
    ds['sample_count'].loc[{'height': 500}] = 10  # invalid (-400)
    ds['sample_count'].loc[{'height': 700}] = 50  # VALID (-200) - has support!
    ds['sample_count'].loc[{'height': 900}] = 50  # VALID
    ds['sample_count'].loc[{'height': 1100}] = 10  # invalid (+200)
    ds['sample_count'].loc[{'height': 1300}] = 10  # invalid (+400)

    ds['corrected_dbz'].loc[{'height': 900}] = 25.0

    original = ds['corrected_dbz'].sel(height=900).copy()
    ds_cleaned = remove_isolated_echo(ds, 900)

    # Should NOT be removed
    xr.testing.assert_identical(ds_cleaned['corrected_dbz'].sel(height=900), original)


def test_remove_isolated_echo_far_neighbor_valid(sample_profile):
    """Test that echo with valid neighbor 2 levels away is not removed."""
    ds = sample_profile.copy(deep=True)

    # Echo at 900m with valid neighbor at +400m
    ds['sample_count'].loc[{'height': 500}] = 10  # invalid (-400)
    ds['sample_count'].loc[{'height': 700}] = 10  # invalid (-200)
    ds['sample_count'].loc[{'height': 900}] = 50  # VALID
    ds['sample_count'].loc[{'height': 1100}] = 10  # invalid (+200)
    ds['sample_count'].loc[{'height': 1300}] = 50  # VALID (+400) - has support!

    ds['corrected_dbz'].loc[{'height': 900}] = 25.0

    original = ds['corrected_dbz'].sel(height=900).copy()
    ds_cleaned = remove_isolated_echo(ds, 900)

    # Should NOT be removed
    xr.testing.assert_identical(ds_cleaned['corrected_dbz'].sel(height=900), original)


def test_remove_isolated_echo_at_edge(sample_profile):
    """Test that remove_isolated_echo handles edge heights gracefully."""
    ds = sample_profile.copy(deep=True)

    # Try at lowest height - should return unchanged
    ds_cleaned = remove_isolated_echo(ds, 100)

    xr.testing.assert_identical(ds_cleaned, ds)


def test_remove_isolated_echo_multiple_time_points(sample_profile):
    """Test isolated echo removal across multiple time points."""
    ds = sample_profile.copy(deep=True)

    # Set up isolated echo at 900m for some time points
    ds['sample_count'].loc[{'height': 500}] = 10
    ds['sample_count'].loc[{'height': 700}] = 10
    ds['sample_count'].loc[{'height': 1100}] = 10
    ds['sample_count'].loc[{'height': 1300}] = 10

    # Some time points are isolated, some have support
    for t in range(10):
        if t < 5:
            ds['sample_count'].loc[{'height': 900, 'time': t}] = 50  # isolated
        else:
            ds['sample_count'].loc[{'height': 900, 'time': t}] = 50
            ds['sample_count'].loc[{'height': 700, 'time': t}] = 50  # has support

    ds['corrected_dbz'].loc[{'height': 900}] = 25.0

    ds_cleaned = remove_isolated_echo(ds, 900)

    # First 5 time points should be MDS, last 5 should be unchanged
    for t in range(5):
        result = ds_cleaned['corrected_dbz'].sel(height=900, time=t).values
        assert result == MDS, f"Time {t} should be MDS"

    for t in range(5, 10):
        result = ds_cleaned['corrected_dbz'].sel(height=900, time=t).values
        assert result == 25.0, f"Time {t} should be unchanged"


def test_smooth_spikes_runs_without_error(sample_profile):
    """Test that the main smooth_spikes function runs without error."""
    ds = sample_profile.copy(deep=True)

    # Add some valid data
    ds['sample_count'].loc[{'height': slice(500, 1500)}] = 50
    ds['lin_dbz'].loc[{'height': slice(500, 1500)}] = 20.0
    ds['corrected_dbz'].loc[{'height': slice(500, 1500)}] = 20.0

    # Should run without error
    ds_smoothed = smooth_spikes(ds)

    assert isinstance(ds_smoothed, xr.Dataset)
    assert 'corrected_dbz' in ds_smoothed
