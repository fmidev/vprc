"""Bright band (melting layer) detection for VPR correction.

The bright band is a layer of enhanced radar reflectivity caused by melting
snowflakes. Detecting it is essential for accurate VPR correction, as the
reflectivity enhancement must be accounted for when extrapolating to ground.

Based on allprof_prodx2.pl section "Bright Band".

Detection approach:
    1. Compute vertical gradients of corrected reflectivity
    2. Look for characteristic pattern: positive gradient below peak,
       negative gradient above (the "BB signature")
    3. Constrain search around freezing level when available
    4. Validate detection using amplitude and gradient thresholds
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import xarray as xr

from .constants import MDS, STEP


@dataclass
class BrightBandResult:
    """Results of bright band detection.

    Attributes:
        detected: True if a valid bright band was found
        peak_height: Height of maximum reflectivity in BB (m above antenna)
        bottom_height: Lower boundary of the melting layer (m)
        top_height: Upper boundary of the melting layer (m)
        peak_dbz: Reflectivity at the bright band peak (dBZ)
        amplitude_below: dBZ difference from bottom to peak
        amplitude_above: dBZ difference from peak to top
        freezing_level_m: Freezing level used for detection (if available)
    """

    detected: bool = False
    peak_height: int | None = None
    bottom_height: int | None = None
    top_height: int | None = None
    peak_dbz: float | None = None
    amplitude_below: float | None = None
    amplitude_above: float | None = None
    freezing_level_m: float | None = None

    @property
    def thickness(self) -> int | None:
        """Bright band thickness in meters, if detected."""
        if self.top_height is not None and self.bottom_height is not None:
            return self.top_height - self.bottom_height
        return None


def _compute_gradient_sum(gradient: xr.DataArray, height: int) -> float:
    """Compute sum of gradients at height and height+200m.

    This is used in the BB detection pattern matching.
    Perl: $isotaulu[$i][$j][7] = $isotaulu[$i][$j][4] + $isotaulu[$i+200][$j][4]

    Args:
        gradient: Gradient DataArray indexed by height
        height: Height at which to compute sum

    Returns:
        Sum of gradient[height] + gradient[height+STEP], or NaN if either missing
    """
    h_next = height + STEP
    heights = gradient["height"].values

    if height not in heights or h_next not in heights:
        return np.nan

    g1 = float(gradient.sel(height=height).values)
    g2 = float(gradient.sel(height=h_next).values)

    if np.isnan(g1) or np.isnan(g2):
        return np.nan

    return g1 + g2


def _find_bb_candidates(
    dbz: xr.DataArray,
    gradient: xr.DataArray,
    heights: np.ndarray,
    max_height: int,
) -> list[int]:
    """Find candidate bright band peak heights.

    The BB signature is identified by:
    - Valid (>MDS) data at 5 consecutive levels centered on candidate
    - Positive gradient sum at candidate (reflectivity increasing from below)
    - Non-positive gradient sum below candidate (relatively flat or decreasing below)

    Args:
        dbz: Corrected reflectivity values
        gradient: Vertical gradient (dBZ per 200m step, not per meter)
        heights: Array of height values
        max_height: Upper limit for BB search

    Returns:
        List of candidate peak heights
    """
    candidates = []
    step = STEP

    for h in heights:
        h = int(h)
        if h + 2 * step > max_height:
            continue

        # Need 5 consecutive valid levels: h-400, h-200, h, h+200, h+400
        required = [h - 2 * step, h - step, h, h + step, h + 2 * step]

        # Check heights exist in array (skip if near boundaries)
        if not all(hr in heights for hr in required):
            continue

        # Check all levels have valid data
        valid = all(float(dbz.sel(height=hr).values) > MDS for hr in required)
        if not valid:
            continue

        # Compute gradient pattern
        # Gradient sum at h and h+step (looking at gradient from h to h+200 and h+200 to h+400)
        grad_sum_here = _compute_gradient_sum(gradient, h)
        grad_sum_below = _compute_gradient_sum(gradient, h - 2 * step)
        grad_sum_below2 = _compute_gradient_sum(gradient, h - step)

        if np.isnan(grad_sum_here):
            continue

        # BB signature: grad sum at h is positive (local max forming),
        # grad sums below are non-positive (increasing toward peak)
        # Threshold of 2.5 from Perl: ($isotaulu[$i][$j][7] + $isotaulu[$i+400][$j][4]) >= 2.5
        grad_at_plus_400 = float(gradient.sel(height=h + 2 * step).values) if h + 2 * step in heights else 0

        if grad_sum_here + grad_at_plus_400 >= 2.5:
            # Check gradients below are non-positive
            if not np.isnan(grad_sum_below) and not np.isnan(grad_sum_below2):
                if grad_sum_below + grad_sum_below2 <= 0:
                    candidates.append(h)

    return candidates


def _find_bb_boundaries(
    dbz: xr.DataArray,
    gradient: xr.DataArray,
    peak_height: int,
    heights: np.ndarray,
) -> tuple[int | None, int | None, float | None, float | None]:
    """Find the bottom and top boundaries of the bright band.

    Uses gradient differences to locate where the BB transitions
    to rain below and snow above.

    Args:
        dbz: Corrected reflectivity values
        gradient: Vertical gradient
        peak_height: Height of the BB peak
        heights: Array of height values

    Returns:
        Tuple of (bottom_height, top_height, amplitude_below, amplitude_above)
    """
    step = STEP
    peak_dbz = float(dbz.sel(height=peak_height).values)

    # Find bottom: look for maximum gradient change below peak
    bottom_height = None
    max_grad_diff_below = 0.0
    amplitude_below = None

    for k in range(peak_height, max(peak_height - 800, 0), -step):
        k_below = k - step
        k_below2 = k - 2 * step

        if k_below not in heights or k_below2 not in heights:
            continue

        g1 = float(gradient.sel(height=k_below2).values) if k_below2 in heights else np.nan
        g2 = float(gradient.sel(height=k_below).values) if k_below in heights else np.nan

        if np.isnan(g1) or np.isnan(g2):
            continue

        grad_diff = g1 - g2

        if grad_diff >= max_grad_diff_below and k_below in heights:
            max_grad_diff_below = grad_diff
            bottom_height = k_below
            dbz_at_bottom = float(dbz.sel(height=k_below).values)
            amplitude_below = peak_dbz - dbz_at_bottom

    # Find top: look for maximum gradient change above peak
    top_height = None
    max_grad_diff_above = 0.0
    amplitude_above = None

    for k in range(peak_height, min(peak_height + 800, int(heights[-1])), step):
        k_above = k + step

        if k not in heights or k_above not in heights:
            continue

        g1 = float(gradient.sel(height=k).values) if k in heights else np.nan
        g2 = float(gradient.sel(height=k_above).values) if k_above in heights else np.nan

        if np.isnan(g1) or np.isnan(g2):
            continue

        grad_diff = g1 - g2

        if grad_diff >= max_grad_diff_above:
            max_grad_diff_above = grad_diff
            top_height = k_above
            dbz_at_top = float(dbz.sel(height=k_above).values)
            amplitude_above = peak_dbz - dbz_at_top

    return bottom_height, top_height, amplitude_below, amplitude_above


def _validate_bright_band(
    result: BrightBandResult,
    gradient: xr.DataArray,
) -> bool:
    """Validate a bright band detection.

    Applies quality checks from the Perl implementation to reject
    false positives.

    Args:
        result: Preliminary BB detection result
        gradient: Vertical gradient for additional checks

    Returns:
        True if the detection passes validation
    """
    if result.top_height is None:
        return False

    # Check thickness is reasonable (not too thin)
    if result.thickness is not None and result.thickness < 400:
        return False

    # Check amplitude above is meaningful
    if result.amplitude_above is not None and result.amplitude_above < 1.0:
        return False

    # Gradient check above BB top should show continued decrease
    # (Perl: if ($isotaulu[$zyla][$j][3] * 200 > $ylaamp / 2) -> reject)
    if result.top_height in gradient["height"].values:
        grad_above = float(gradient.sel(height=result.top_height).values)
        if not np.isnan(grad_above) and result.amplitude_above is not None:
            # grad_above is dBZ/step; if positive and > half the amplitude, reject
            if grad_above * STEP > result.amplitude_above / 2:
                return False

    return True


def compute_dbz_gradient_per_step(ds: xr.Dataset) -> xr.DataArray:
    """Compute vertical gradient of corrected_dbz in dBZ per step.

    Unlike the gradient in clutter.py (which is dBZ/m), this returns
    the difference between adjacent levels (dBZ per 200m step),
    matching the Perl BB detection logic.

    Args:
        ds: Dataset with 'corrected_dbz' variable

    Returns:
        DataArray with gradient values, indexed at lower height of each pair
    """
    dbz = ds["corrected_dbz"]

    # Compute: gradient[i] = dbz[i] - dbz[i+1] (positive = decreasing upward)
    # Using shift to get dbz at height+step
    dbz_shifted = dbz.shift(height=-1)
    gradient = dbz - dbz_shifted

    # Mask where either value is missing
    valid = (dbz > MDS) & (dbz_shifted > MDS)
    gradient = gradient.where(valid)

    return gradient


def detect_bright_band(
    ds: xr.Dataset,
    layer_top: int | None = None,
) -> BrightBandResult:
    """Detect bright band in a vertical profile.

    Searches for the characteristic melting layer signature in the
    corrected reflectivity profile. If freezing_level_m is available
    in the dataset attributes, the search is constrained around it.

    Args:
        ds: xarray Dataset with 'corrected_dbz', indexed by 'height'.
            Should have 'freezing_level_m' in attrs (can be None).
        layer_top: Upper limit of search (e.g., top of lowest layer).
            If None, uses top of profile.

    Returns:
        BrightBandResult with detection status and parameters

    Example:
        >>> ds = read_vvp("profile.txt")
        >>> ds = apply_clutter_correction(ds)
        >>> ds = apply_spike_smoothing(ds)
        >>> bb = detect_bright_band(ds)
        >>> if bb.detected:
        ...     print(f"BB peak at {bb.peak_height}m")
    """
    freezing_level = ds.attrs.get("freezing_level_m", None)
    heights = ds["height"].values
    dbz = ds["corrected_dbz"]

    # Determine search bounds
    if layer_top is None:
        layer_top = int(heights[-1])

    # Skip detection if freezing level is not positive or layer too low
    if freezing_level is not None and freezing_level <= 0:
        return BrightBandResult(
            detected=False, freezing_level_m=freezing_level
        )

    if layer_top < 1300:
        return BrightBandResult(
            detected=False, freezing_level_m=freezing_level
        )

    # Compute gradient (dBZ per step, not per meter)
    gradient = compute_dbz_gradient_per_step(ds)

    # Find candidate peaks
    candidates = _find_bb_candidates(dbz, gradient, heights, layer_top)

    if not candidates:
        return BrightBandResult(
            detected=False, freezing_level_m=freezing_level
        )

    # Select best candidate based on proximity to freezing level
    best_candidate = None
    min_distance = float("inf")

    for h in candidates:
        if freezing_level is not None and freezing_level > 0:
            distance = abs(h - freezing_level)
            if distance < min_distance:
                min_distance = distance
                best_candidate = h
        else:
            # Without freezing level, take highest candidate (most likely)
            if best_candidate is None or h > best_candidate:
                best_candidate = h

    if best_candidate is None:
        return BrightBandResult(
            detected=False, freezing_level_m=freezing_level
        )

    # Find boundaries
    bottom, top, amp_below, amp_above = _find_bb_boundaries(
        dbz, gradient, best_candidate, heights
    )

    peak_dbz = float(dbz.sel(height=best_candidate).values)

    result = BrightBandResult(
        detected=True,
        peak_height=best_candidate,
        bottom_height=bottom,
        top_height=top,
        peak_dbz=peak_dbz,
        amplitude_below=amp_below,
        amplitude_above=amp_above,
        freezing_level_m=freezing_level,
    )

    # Validate
    if not _validate_bright_band(result, gradient):
        result.detected = False

    return result
