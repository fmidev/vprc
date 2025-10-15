#!/usr/bin/env python3
"""
Ground clutter detection and removal for VPR correction.

Based on allprof_prodx2.pl lines 370-488 (gradient calculation and clutter removal).
"""

import numpy as np
import xarray as xr

from .constants import MDS, GMDS, MKKYNNYS, MIN_SAMPLES, FREEZING_LEVEL_MIN, STEP


def compute_gradient(ds: xr.Dataset, min_samples: int = MIN_SAMPLES) -> xr.DataArray:
    """
    Compute vertical gradient of corrected_dbz with quality checking.

    Based on allprof_prodx2.pl lines 370-382.

    The Perl implementation calculates forward differences: gradient[i] = (dbz[i+200] - dbz[i]) / 200
    This is critical for ground clutter detection logic.

    Args:
        ds: xarray Dataset with 'corrected_dbz' and 'count' variables
        min_samples: Minimum sample count for valid gradient (default: 30)

    Returns:
        DataArray with vertical gradient in dBZ/m.
        Invalid gradients (insufficient samples) are marked as NaN.
        Gradient[i] represents the slope from height[i] to height[i+1].

    Note:
        Uses forward differences, not centered differences, to match Perl logic.
    """
    dbz = ds['corrected_dbz']
    heights = ds['height']

    # Compute forward difference: (dbz[i+1] - dbz[i]) / (height[i+1] - height[i])
    dbz_diff = dbz.diff('height')  # dbz[i+1] - dbz[i]
    height_diff = heights.diff('height')  # height[i+1] - height[i]
    gradient = dbz_diff / height_diff

    # The diff operation shifts the index, so gradient[i] now represents
    # the gradient FROM height[i] TO height[i+1]
    # We need to align this back to the original height coordinate
    # After diff, the coordinate is the upper bound, but we want lower bound
    gradient = gradient.assign_coords(height=heights[:-1].values)

    # Reindex to match original height coordinate, filling missing with NaN
    gradient = gradient.reindex(height=heights, fill_value=float('nan'))

    # Mark invalid where sample count is too low
    # For forward difference gradient[i], we need samples at both i and i+1
    valid_samples = ds['count'] >= min_samples
    valid_samples_next = valid_samples.shift(height=-1, fill_value=False)
    valid_gradient = valid_samples & valid_samples_next

    # Apply mask - set invalid gradients to NaN
    gradient = gradient.where(valid_gradient)

    return gradient


def remove_ground_clutter(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove ground clutter contamination from vertical profiles.

    Based on allprof_prodx2.pl lines 393-488.

    Algorithm:
        Ground clutter creates artificially high reflectivity near the surface
        with steep negative vertical gradients. The algorithm:

        1. Checks freezing level - skips correction if 0 < FL < 1000m
        2. Examines lowest 3-4 height levels for steep negative gradients
        3. Extrapolates clean values downward from first uncontaminated level
        4. Protects against over-correction (keeps original if correction is higher)
        5. Removes all data if gradients are completely missing

    Detection criteria (all must be met):
        - Echo present: dBZ > MDS (-45)
        - Valid gradient: not NaN
        - Steep negative gradient: gradient < MKKYNNYS (-0.005 dBZ/m = -1 dBZ/200m)

    Args:
        ds: xarray Dataset with 'corrected_dbz', 'count', and metadata

    Returns:
        Dataset with corrected_dbz modified to remove ground clutter.
        Original Dataset is not modified (copy is returned).

    Note:
        This implementation uses vectorized numpy/xarray operations instead of
        the nested loops in the Perl code, but produces equivalent results.
    """
    # Work on a copy to avoid modifying input
    ds = ds.copy(deep=True)

    # Get freezing level from metadata
    freezing_level = ds.attrs.get('freezing_level_m', None)

    # Skip correction if freezing level is between 0 and 1000m (line 392)
    # (but proceed if FL is None, 0, or > 1000m)
    if freezing_level is not None and 0 < freezing_level < FREEZING_LEVEL_MIN:
        return ds

    # Get lowest level from metadata
    lowest_level = ds.attrs.get('lowest_level_offset_m', 100)

    # Compute gradient with quality checks
    gradient = compute_gradient(ds)

    # Get heights array
    heights = ds['height'].values

    # Find the actual lowest level in the data to process
    # The Perl code uses $alintaso which is the lowest_level_offset_m value
    # But the actual data starts at the first height level (usually 100m)
    # The Perl loop: for ( $i = $alintaso ; $i < 200 ; $i = $i + 200 )
    # with $alintaso typically 100-200, this runs once for the first data level

    # Use the minimum height in the dataset as the base level
    i = int(heights.min())

    if i not in heights:
        return ds

    # Define the heights we'll work with
    h0 = i           # e.g., 100m
    h1 = i + 200     # e.g., 300m
    h2 = i + 400     # e.g., 500m
    h3 = i + 600     # e.g., 700m

    # Helper function to check clutter signature at a height
    def has_clutter(height):
        if height not in heights:
            return False
        dbz = ds['corrected_dbz'].sel(height=height)
        grad = gradient.sel(height=height)
        result = (dbz > MDS) & ~np.isnan(grad) & (grad < MKKYNNYS)
        # Extract the scalar boolean value from the DataArray
        return bool(result.values)

    # Helper function to get gradient or check if missing
    def gradient_missing(height):
        if height not in heights:
            return True
        return bool(np.isnan(gradient.sel(height=height)).all())

    # Check for clutter at each level
    clutter_h0 = has_clutter(h0) if h0 in heights else False
    clutter_h1 = has_clutter(h1) if h1 in heights else False
    clutter_h2 = has_clutter(h2) if h2 in heights else False

    # Case 1: Clutter only at h0 (lines 397-404)
    if clutter_h0 and h1 in heights:
        dbz_ref = ds['corrected_dbz'].sel(height=h1)
        corrected = dbz_ref - MKKYNNYS * 200
        original = ds['corrected_dbz'].sel(height=h0)
        ds['corrected_dbz'].loc[dict(height=h0)] = xr.where(
            corrected < original, corrected, original
        )

    # Case 2: Clutter at h1 (and possibly h0) (lines 407-429)
    if clutter_h1 and h2 in heights:
        dbz_ref = ds['corrected_dbz'].sel(height=h2)

        # Correct h1
        if h1 in heights:
            corrected_h1 = dbz_ref - MKKYNNYS * 200
            original_h1 = ds['corrected_dbz'].sel(height=h1)
            ds['corrected_dbz'].loc[dict(height=h1)] = corrected_h1

        # Correct h0
        if h0 in heights:
            corrected_h0 = dbz_ref - 2 * MKKYNNYS * 200
            original_h0 = ds['corrected_dbz'].sel(height=h0)
            # Protection: don't increase value
            ds['corrected_dbz'].loc[dict(height=h0)] = xr.where(
                corrected_h0 < original_h0, corrected_h0, original_h0
            )

    # Case 3: Clutter at h2 (and possibly h1, h0) (lines 432-468)
    if clutter_h2 and h3 in heights:
        dbz_ref = ds['corrected_dbz'].sel(height=h3)

        # Correct h2
        if h2 in heights:
            corrected_h2 = dbz_ref - MKKYNNYS * 200
            ds['corrected_dbz'].loc[dict(height=h2)] = corrected_h2

        # Correct h1 (note: Perl has typo setting h1 twice, line 444-445)
        if h1 in heights:
            corrected_h1 = dbz_ref - 2 * MKKYNNYS * 200
            original_h1 = ds['corrected_dbz'].sel(height=h1)
            # Protection: don't increase value (lines 450-457)
            if not gradient_missing(h1):
                ds['corrected_dbz'].loc[dict(height=h1)] = xr.where(
                    corrected_h1 < original_h1, corrected_h1, original_h1
                )

        # Correct h0
        if h0 in heights:
            corrected_h0 = dbz_ref - 3 * MKKYNNYS * 200
            original_h0 = ds['corrected_dbz'].sel(height=h0)
            # Protection: don't increase value (lines 460-465)
            if not gradient_missing(h0):
                ds['corrected_dbz'].loc[dict(height=h0)] = xr.where(
                    corrected_h0 < original_h0, corrected_h0, original_h0
                )

    # Special case: no data at h2 means set h0 to MDS (lines 469-471)
    if h2 in heights:
        if ds['corrected_dbz'].sel(height=h2) <= MDS:
            if h0 in heights:
                ds['corrected_dbz'].loc[dict(height=h0)] = MDS

    # Handle complete clutter removal when gradients are missing (lines 473-487)
    if h0 in heights and gradient_missing(h0):
        ds['corrected_dbz'].loc[dict(height=h0)] = MDS

    if h1 in heights and gradient_missing(h1):
        ds['corrected_dbz'].loc[dict(height=h0)] = MDS
        ds['corrected_dbz'].loc[dict(height=h1)] = MDS

    if h2 in heights and gradient_missing(h2):
        ds['corrected_dbz'].loc[dict(height=h0)] = MDS
        ds['corrected_dbz'].loc[dict(height=h1)] = MDS
        ds['corrected_dbz'].loc[dict(height=h2)] = MDS

    return ds
