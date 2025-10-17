"""Spike smoothing operations for VVP profiles.

Based on the "piikkien tasoitus" section of allprof_prodx2.pl.
These functions remove or smooth unphysical spikes and isolated echoes
in vertical reflectivity profiles.
"""

import xarray as xr
import numpy as np
from .constants import (
    SPIKE_AMPLITUDE_THRESHOLD,
    LARGE_POSITIVE_SPIKE_THRESHOLD,
)


def correct_lower_boundary_spike(
    ds: xr.Dataset,
    height: int,
    sample_threshold: int = 30,
) -> xr.Dataset:
    """Remove false positive spikes at the bottom of echo regions.

    Examines levels where there is missing data below but valid data at the current
    level and two levels above. If reflectivity increases at this boundary (unphysical),
    corrects the spike by either:
    - Extrapolating downward using the positive gradient above, or
    - Setting the spike value equal to the level above

    Based on allprof_prodx2.pl ("alarajan positiivinen piikki").

    Parameters
    ----------
    ds : xr.Dataset
        VVP dataset with 'lin_dbz' and 'corrected_dbz' variables
        Must have 'height' and 'time' dimensions
    height : int
        The height level (in meters) to check for boundary spikes
    sample_threshold : int, default=30
        Minimum number of samples required for valid data (Perl $kynnys)

    Returns
    -------
    xr.Dataset
        Dataset with corrected_dbz modified in-place for the specified height

    Notes
    -----
    The algorithm checks the pattern:
    - height - 200m: missing (samples < threshold)
    - height: valid data
    - height + 200m: valid data
    - height + 400m: valid data

    If reflectivity at height+200m exceeds corrected value at height+400m
    (unphysical increase at boundary), apply correction.
    """
    # Check if all required height levels exist
    h_minus_200 = height - 200
    h_plus_200 = height + 200
    h_plus_400 = height + 400
    h_plus_600 = height + 600

    required_heights = [h_minus_200, height, h_plus_200, h_plus_400, h_plus_600]
    if not all(h in ds.height.values for h in required_heights):
        return ds

    # Extract data at relevant heights
    samples_below = ds['sample_count'].sel(height=h_minus_200)
    samples_current = ds['sample_count'].sel(height=height)
    samples_plus_200 = ds['sample_count'].sel(height=h_plus_200)
    samples_plus_400 = ds['sample_count'].sel(height=h_plus_400)

    lin_dbz_plus_200 = ds['lin_dbz'].sel(height=h_plus_200)
    corrected_plus_400 = ds['corrected_dbz'].sel(height=h_plus_400)
    corrected_plus_600 = ds['corrected_dbz'].sel(height=h_plus_600)

    # Identify where the pattern matches:
    # - Missing data below (samples < threshold)
    # - Valid data at current level and two levels above
    pattern_match = (
        (samples_below < sample_threshold) &
        (samples_current > sample_threshold) &
        (samples_plus_200 > sample_threshold) &
        (samples_plus_400 > sample_threshold)
    )

    # Check for unphysical increase at boundary
    # (Perl: if reflectivity decreases at edge - epäfysikaalista)
    unphysical_increase = lin_dbz_plus_200 - corrected_plus_400 > 0

    # Apply correction where both conditions are met
    needs_correction = pattern_match & unphysical_increase

    if needs_correction.any():
        # Check gradient above: if positive, extrapolate downward
        gradient_above = corrected_plus_600 - corrected_plus_400
        positive_gradient = gradient_above > 0

        # Two correction strategies
        # Strategy 1: Extrapolate using positive gradient above
        extrapolated = corrected_plus_400 - gradient_above

        # Strategy 2: Set equal to level above
        equal_to_above = corrected_plus_400

        # Apply appropriate correction
        correction = xr.where(
            positive_gradient,
            extrapolated,
            equal_to_above
        )

        # Update corrected_dbz at height+200m where correction is needed
        ds['corrected_dbz'].loc[{'height': h_plus_200}] = xr.where(
            needs_correction,
            correction,
            ds['corrected_dbz'].sel(height=h_plus_200)
        )

    return ds


def correct_upper_boundary_spike(
    ds: xr.Dataset,
    height: int,
    sample_threshold: int = 30,
) -> xr.Dataset:
    """Remove false positive spikes at the top of echo regions.

    Examines levels where there is valid data at the current and one level above,
    but missing data at two levels above. If reflectivity increases at this boundary
    (unphysical), corrects the spike by either:
    - Extrapolating upward using the negative gradient below, or
    - Setting the spike value equal to the level below

    Based on allprof_prodx2.pl lines 458-487 ("ylärajan positiivinen piikki").

    Parameters
    ----------
    ds : xr.Dataset
        VVP dataset with 'lin_dbz', 'corrected_dbz', and 'sample_count' variables
        Must have 'height' and 'time' dimensions
    height : int
        The height level (in meters) to check for boundary spikes
    sample_threshold : int, default=30
        Minimum number of samples required for valid data (Perl $kynnys)

    Returns
    -------
    xr.Dataset
        Dataset with corrected_dbz modified in-place for the specified height

    Notes
    -----
    The algorithm checks the pattern:
    - height: valid data
    - height + 200m: valid data
    - height + 400m: missing (samples < threshold)
    - height + 600m: missing (samples < threshold)

    If reflectivity at height+200m exceeds corrected value at height
    (unphysical increase at boundary), apply correction.
    """
    # Check if all required height levels exist
    h_minus_200 = height - 200
    h = height
    h_plus_200 = height + 200
    h_plus_400 = height + 400
    h_plus_600 = height + 600

    required_heights = [h_minus_200, h, h_plus_200, h_plus_400, h_plus_600]
    if not all(hgt in ds.height.values for hgt in required_heights):
        return ds

    # Extract data at relevant heights
    samples_current = ds['sample_count'].sel(height=h)
    samples_plus_200 = ds['sample_count'].sel(height=h_plus_200)
    samples_plus_400 = ds['sample_count'].sel(height=h_plus_400)
    samples_plus_600 = ds['sample_count'].sel(height=h_plus_600)

    lin_dbz_plus_200 = ds['lin_dbz'].sel(height=h_plus_200)
    corrected_current = ds['corrected_dbz'].sel(height=h)
    corrected_minus_200 = ds['corrected_dbz'].sel(height=h_minus_200)

    # Identify where the pattern matches:
    # - Valid data at current and one level above
    # - Missing data at two levels above
    pattern_match = (
        (samples_current > sample_threshold) &
        (samples_plus_200 > sample_threshold) &
        (samples_plus_400 < sample_threshold) &
        (samples_plus_600 < sample_threshold)
    )

    # Check for unphysical increase at boundary
    # Perl: if reflectivity increases at edge (kaiku kasvaa reunalle)
    unphysical_increase = lin_dbz_plus_200 - corrected_current > 0

    # Apply correction where both conditions are met
    needs_correction = pattern_match & unphysical_increase

    if needs_correction.any():
        # Check gradient below: if negative, extrapolate upward
        gradient_below = corrected_current - corrected_minus_200
        negative_gradient = gradient_below < 0

        # Two correction strategies
        # Strategy 1: Extrapolate using negative gradient below
        # corrected[i+200] = corrected[i] + (corrected[i] - corrected[i-200])
        extrapolated = corrected_current + gradient_below

        # Strategy 2: Set equal to level below
        equal_to_below = corrected_current

        # Apply appropriate correction
        correction = xr.where(
            negative_gradient,
            extrapolated,
            equal_to_below
        )

        # Update corrected_dbz at height+200m where correction is needed
        ds['corrected_dbz'].loc[{'height': h_plus_200}] = xr.where(
            needs_correction,
            correction,
            ds['corrected_dbz'].sel(height=h_plus_200)
        )

    return ds


def smooth_positive_spikes(
    ds: xr.Dataset,
    sample_threshold: int = 30,
    spike_threshold: float = SPIKE_AMPLITUDE_THRESHOLD,
    large_spike_threshold: float = LARGE_POSITIVE_SPIKE_THRESHOLD,
) -> xr.Dataset:
    """Smooth positive spikes using 3-point rolling mean (with rolling windows).

    Detects and smooths positive spikes where the middle point in a 3-level
    sequence exceeds both neighbors by more than the spike threshold.
    For moderate spikes, applies 3-point moving average. For large spikes,
    replaces the spike with the average of the two neighbors only.

    This operation is ideal for rolling window implementation because it uses
    a symmetric 3-point window applied uniformly across all valid data.

    Based on allprof_prodx2.pl lines 486-520 ("Positiivinen piikki").

    Parameters
    ----------
    ds : xr.Dataset
        VVP dataset with 'lin_dbz', 'corrected_dbz', and 'sample_count' variables
        Must have 'height' and 'time' dimensions
    sample_threshold : int, default=30
        Minimum number of samples required for valid data (Perl $kynnys)
    spike_threshold : float, default=3.0
        Amplitude threshold for spike detection (dBZ)
        A point is a spike if it exceeds both neighbors by this amount
    large_spike_threshold : float, default=4.0
        Threshold for using 2-point vs 3-point average (dBZ, Perl $dbzkynnys1)
        If difference between original and 3-point average exceeds this,
        use 2-point average (neighbors only) instead

    Returns
    -------
    xr.Dataset
        Dataset with smoothed positive spikes in corrected_dbz

    Notes
    -----
    Algorithm:
    1. Check three consecutive valid levels (i, i+1, i+2)
    2. If middle level exceeds both neighbors by >spike_threshold:
       a. Calculate 3-point moving average
       b. Calculate difference between original and smoothed
       c. If difference > large_spike_threshold: use 2-point average
       d. Otherwise: use 3-point average

    The Perl code uses lin_dbz (original) for spike detection but modifies
    corrected_dbz (working values), so we follow the same approach.
    """
    result = ds.copy(deep=True)

    # Create mask for valid data based on sample count
    is_valid = result['sample_count'] >= sample_threshold

    # We need 3 consecutive valid levels
    # Use shift to align data from i, i+1, i+2 positions
    valid_current = is_valid  # level i
    valid_above_1 = is_valid.shift(height=1, fill_value=False)  # level i+1
    valid_above_2 = is_valid.shift(height=2, fill_value=False)  # level i+2

    # All three levels must be valid
    three_valid = valid_current & valid_above_1 & valid_above_2

    # Get original dBZ values at the three positions (for spike detection)
    # Perl uses lin_dbz (isotaulu[i][j][0]) for detection
    lin_dbz = result['lin_dbz']
    lin_current = lin_dbz  # level i
    lin_above_1 = lin_dbz.shift(height=1, fill_value=0)  # level i+1
    lin_above_2 = lin_dbz.shift(height=2, fill_value=0)  # level i+2

    # Detect positive spikes at middle position (i+1):
    # lin_dbz[i+1] - lin_dbz[i] > 3 AND lin_dbz[i+1] - lin_dbz[i+2] > 3
    # When viewed from position i, we check if i+1 exceeds both i and i+2
    spike_above_lower = lin_above_1 - lin_current > spike_threshold
    spike_above_upper = lin_above_1 - lin_above_2 > spike_threshold

    is_positive_spike = three_valid & spike_above_lower & spike_above_upper

    # If no spikes detected, return early
    if not is_positive_spike.any():
        return result

    # Calculate corrections using corrected_dbz values
    corrected = result['corrected_dbz']
    corr_current = corrected  # level i
    corr_above_1 = corrected.shift(height=1, fill_value=0)  # level i+1
    corr_above_2 = corrected.shift(height=2, fill_value=0)  # level i+2

    # Calculate 3-point moving average
    three_point_avg = (corr_current + corr_above_1 + corr_above_2) / 3

    # Calculate difference between original and 3-point average at spike position
    # Perl first applies 3-point average, THEN checks if difference is too large
    # Perl: $dbzero = $isotaulu[$i+200][$j][0] - $isotaulu[$i+200][$j][5]
    # where [5] is now the 3-point average
    diff_after_smoothing = lin_above_1 - three_point_avg

    # For large spikes (where even 3-point average leaves big difference),
    # use 2-point average (neighbors only, exclude spike completely)
    two_point_avg = (corr_current + corr_above_2) / 2

    # Choose correction method based on spike magnitude
    # Use 2-point if the residual after 3-point smoothing is still > threshold
    use_two_point = diff_after_smoothing > large_spike_threshold
    correction = xr.where(use_two_point, two_point_avg, three_point_avg)

    # Apply correction at the spike position (which is shifted up by 1)
    # We need to shift the spike mask and correction down by 1 to apply at correct position
    is_spike_at_this_height = is_positive_spike.shift(height=-1, fill_value=False)
    correction_at_this_height = correction.shift(height=-1, fill_value=0)

    result['corrected_dbz'] = xr.where(
        is_spike_at_this_height,
        correction_at_this_height,
        result['corrected_dbz']
    )

    return result


def smooth_spikes(ds: xr.Dataset) -> xr.Dataset:
    """Apply all spike smoothing operations to a VVP profile.

    This is the main entry point that applies all 6 spike smoothing operations
    from allprof_prodx2.pl in the correct sequence.

    Parameters
    ----------
    ds : xr.Dataset
        VVP dataset with 'lin_dbz', 'corrected_dbz', and 'sample_count' variables
        Must have 'height' and 'time' dimensions

    Returns
    -------
    xr.Dataset
        Dataset with smoothed corrected_dbz values

    Notes
    -----
    Processing order:
    1. Lower boundary spike correction
    2. Upper boundary spike correction
    3. Positive spike smoothing
    4. Negative spike smoothing
    5. Gap filling
    6. Isolated echo removal
    """
    # Get lowest processing height (after clutter removal)
    # For now, start from the lowest available height
    # TODO: This should come from clutter removal module
    heights = sorted(ds.height.values)
    start_height = heights[0]

    # Process each height level for operations 1 and 2
    for height in heights:
        if height < start_height:
            continue

        # Operation 1: Lower boundary spike correction
        ds = correct_lower_boundary_spike(ds, height)

        # Operation 2: Upper boundary spike correction
        ds = correct_upper_boundary_spike(ds, height)

    # Operation 3: Positive spike smoothing (rolling window - processes all heights)
    ds = smooth_positive_spikes(ds)

    # TODO: Add remaining operations
    # Operation 4: Negative spike smoothing
    # Operation 5: Gap filling
    # Operation 6: Isolated echo removal

    return ds
