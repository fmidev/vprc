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
    LARGE_NEGATIVE_SPIKE_THRESHOLD,
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


def _smooth_spikes_rolling(
    ds: xr.Dataset,
    spike_type: str,
    sample_threshold: int = 30,
    spike_threshold: float = SPIKE_AMPLITUDE_THRESHOLD,
    large_spike_threshold: float | None = None,
) -> xr.Dataset:
    """Internal function to smooth positive or negative spikes using rolling windows.

    This is a unified implementation for operations 3 and 4, which differ only in:
    - Detection direction (> for positive, < for negative)
    - Large spike threshold (4 for positive, -10 for negative)

    Parameters
    ----------
    ds : xr.Dataset
        VVP dataset with corrected_dbz and sample_count variables
    spike_type : str
        Either 'positive' or 'negative'
    sample_threshold : int
        Minimum samples for valid data
    spike_threshold : float
        Amplitude threshold for spike detection (3.0 dBZ in Perl)
    large_spike_threshold : float, optional
        Threshold for 2-point vs 3-point average
        If None, uses LARGE_POSITIVE_SPIKE_THRESHOLD or LARGE_NEGATIVE_SPIKE_THRESHOLD

    Returns
    -------
    xr.Dataset
        Dataset with smoothed spikes
    """
    if spike_type not in ('positive', 'negative'):
        raise ValueError(f"spike_type must be 'positive' or 'negative', got {spike_type}")

    if large_spike_threshold is None:
        large_spike_threshold = (
            LARGE_POSITIVE_SPIKE_THRESHOLD if spike_type == 'positive'
            else LARGE_NEGATIVE_SPIKE_THRESHOLD
        )

    result = ds.copy(deep=True)

    # Identify valid data
    is_valid = result['sample_count'] >= sample_threshold

    # Check if we have 3 consecutive valid levels
    valid_window = is_valid.rolling(height=3, center=True).sum() == 3

    corrected = result['corrected_dbz']

    # Calculate 3-point rolling mean
    smoothed = corrected.rolling(height=3, center=True, min_periods=3).mean()

    # Get values at neighbors for spike detection
    val_below = corrected.shift(height=-1)
    val_current = corrected
    val_above = corrected.shift(height=1)

    # Detect spikes based on type
    if spike_type == 'positive':
        # Positive spike: middle point is > threshold above both neighbors
        spike_above_below = (val_current - val_below > spike_threshold)
        spike_above_above = (val_current - val_above > spike_threshold)
        is_spike = valid_window & spike_above_below & spike_above_above

        # Large positive spike: residual after smoothing is still > threshold
        diff_from_smoothed = val_current - smoothed
        is_large_spike = diff_from_smoothed > large_spike_threshold
    else:  # negative
        # Negative spike: middle point is < threshold below both neighbors
        spike_below_below = (val_current - val_below < -spike_threshold)
        spike_below_above = (val_current - val_above < -spike_threshold)
        is_spike = valid_window & spike_below_below & spike_below_above

        # Large negative spike: residual after smoothing is still < threshold
        diff_from_smoothed = val_current - smoothed
        is_large_spike = diff_from_smoothed < large_spike_threshold

    # Two-point average (neighbors only, exclude center)
    two_point_avg = (val_below + val_above) / 2

    # Choose correction based on spike magnitude
    correction = xr.where(is_large_spike, two_point_avg, smoothed)

    result['corrected_dbz'] = xr.where(
        is_spike,
        correction,
        result['corrected_dbz']
    )

    return result


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
    return _smooth_spikes_rolling(
        ds,
        spike_type='positive',
        sample_threshold=sample_threshold,
        spike_threshold=spike_threshold,
        large_spike_threshold=large_spike_threshold,
    )


def smooth_negative_spikes(
    ds: xr.Dataset,
    sample_threshold: int = 30,
    spike_threshold: float = SPIKE_AMPLITUDE_THRESHOLD,
    large_spike_threshold: float = LARGE_NEGATIVE_SPIKE_THRESHOLD,
) -> xr.Dataset:
    """Smooth negative spikes using 3-point rolling mean (with rolling windows).

    Detects and smooths negative spikes where the middle point in a 3-level
    sequence is below both neighbors by more than the spike threshold.
    For moderate spikes, applies 3-point moving average. For large spikes,
    replaces the spike with the average of the two neighbors only.

    This operation is ideal for rolling window implementation because it uses
    a symmetric 3-point window applied uniformly across all valid data.

    Based on allprof_prodx2.pl lines 522-548 ("Negatiivinen piikki").

    Parameters
    ----------
    ds : xr.Dataset
        VVP dataset with 'lin_dbz', 'corrected_dbz', and 'sample_count' variables
        Must have 'height' and 'time' dimensions
    sample_threshold : int, default=30
        Minimum number of samples required for valid data (Perl $kynnys)
    spike_threshold : float, default=3.0
        Amplitude threshold for spike detection (dBZ)
        A point is a spike if it is below both neighbors by this amount
    large_spike_threshold : float, default=-10.0
        Threshold for using 2-point vs 3-point average (dBZ, Perl $dbzkynnys2)
        If difference between original and 3-point average is below this,
        use 2-point average (neighbors only) instead

    Returns
    -------
    xr.Dataset
        Dataset with smoothed negative spikes in corrected_dbz

    Notes
    -----
    Algorithm:
    1. Check three consecutive valid levels (i, i+1, i+2)
    2. If middle level is <spike_threshold below both neighbors:
       a. Calculate 3-point moving average
       b. Calculate difference between original and smoothed
       c. If difference < large_spike_threshold: use 2-point average
       d. Otherwise: use 3-point average

    The Perl code uses lin_dbz (original) for spike detection but modifies
    corrected_dbz (working values), so we follow the same approach.
    """
    return _smooth_spikes_rolling(
        ds,
        spike_type='negative',
        sample_threshold=sample_threshold,
        spike_threshold=spike_threshold,
        large_spike_threshold=large_spike_threshold,
    )


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

    # Operation 4: Negative spike smoothing (rolling window - processes all heights)
    ds = smooth_negative_spikes(ds)

    # TODO: Add remaining operations
    # Operation 5: Gap filling
    # Operation 6: Isolated echo removal

    return ds
