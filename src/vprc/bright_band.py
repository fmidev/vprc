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

import logging
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
import xarray as xr

from .clutter import compute_gradient
from .constants import (
    BB_SPIKE_RESTORATION_OFFSETS,
    LARGE_JUMP_THRESHOLD_DB,
    LOW_BB_HEIGHT_M,
    MDS,
    NEAR_SURFACE_BB_MIN_ZCOUNT,
    NEAR_SURFACE_BB_MIN_ZCOUNT_RATIO,
    NO_BB_CLUTTER_FREEZING_LEVEL_M,
    POST_BB_CLUTTER_HEIGHT_M,
    POST_BB_CLUTTER_THRESHOLD_DB,
    STEP,
)

logger = logging.getLogger(__name__)


class BBRejectionReason(Enum):
    """Reason for bright band detection rejection.

    Used to determine which post-BB clutter correction to apply.
    """

    NOT_REJECTED = "not_rejected"
    NOT_PRECIPITATION = "not_precipitation"
    NO_CANDIDATES = "no_candidates"
    LAYER_TOO_LOW = "layer_too_low"
    FREEZING_LEVEL_INVALID = "freezing_level_invalid"
    VALIDATION_FAILED = "validation_failed"
    NEAR_SURFACE_ZCOUNT = "near_surface_zcount"  # Triggers clutter correction
    EVAPORATION = "evaporation"


@dataclass
class BrightBandResult:
    """Results of bright band detection.

    Attributes:
        detected: True if a valid bright band was found
        peak_height: Discrete BB peak used for restoration (legacy bbalku)
        bottom_height: Lower boundary (m above antenna), or -1 for a surface BB
        top_height: Upper boundary of the melting layer (m)
        peak_dbz: Reflectivity at the bright band peak (dBZ)
        amplitude_below: dBZ difference from bottom to peak
        amplitude_above: dBZ difference from peak to top
        freezing_level_m: Freezing level used for detection (if available)
        rejection_reason: Why BB was rejected (if not detected)
        clutter_correction_applied: True if post-BB clutter correction was applied
        refined_peak_height: Reflectivity-weighted peak height (legacy bbhuippu)
    """

    detected: bool = False
    peak_height: int | None = None
    bottom_height: int | None = None
    top_height: int | None = None
    peak_dbz: float | None = None
    amplitude_below: float | None = None
    amplitude_above: float | None = None
    freezing_level_m: float | None = None
    rejection_reason: BBRejectionReason = BBRejectionReason.NOT_REJECTED
    clutter_correction_applied: bool = False
    refined_peak_height: float | None = None

    @property
    def thickness(self) -> int | None:
        """Bright band thickness in meters, if detected."""
        if self.top_height is not None and self.bottom_height is not None:
            return self.top_height - max(0, self.bottom_height)
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
    - Valid (>MDS) data around the candidate; below-ground levels are optional
    - Raw reflectivity dropping by at least 2.5 dB over 600 m above the candidate
    - Raw reflectivity 400 m below the candidate not exceeding the candidate's

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

        # A surface melting peak does not require below-ground observations.
        if not all(hr in heights or hr < 0 for hr in required):
            continue

        # Check all levels have valid data
        valid = all(hr < 0 or float(dbz.sel(height=hr).values) > MDS for hr in required)
        if not valid:
            continue

        # gradient[k] is the raw drop d(k) = Z(k) - Z(k+200). Perl tests
        # `$isotaulu[$i][$j][7] + $isotaulu[$i+400][$j][4] >= 2.5` above and
        # `[7][$i-200] + [7][$i-400] <= 0` below, with [7][k] = d(k) + d(k+200).
        # As Perl ran, [7] was summed before d(k+200) was set, so it held only
        # d(k). Neither reading is used as such:
        # - Above: the full drop Z(h) - Z(h+600) >= 2.5 dB, as written. As run,
        #   Perl skipped d(h+200) and missed bands whose drop is in that step.
        # - Below: Z(h-400) <= Z(h), as run. As written, the test counted
        #   d(h-200) twice and d(h) once, so a strong drop above the peak
        #   rejected exactly the most distinct bands.
        drop_above = _compute_gradient_sum(gradient, h)
        if np.isnan(drop_above):
            continue
        # Without echo at h+600 the last step is unknown; Perl reads it as 0.
        top_step = (
            float(gradient.sel(height=h + 2 * step).values)
            if h + 2 * step in heights
            else np.nan
        )
        if np.isfinite(top_step):
            drop_above += top_step
        rise_below = sum(
            float(gradient.sel(height=k).values)
            for k in (h - 2 * step, h - step)
            if k >= 0  # A surface peak does not need below-ground levels.
        )

        if drop_above >= 2.5 and rise_below <= 0:
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
        dbz: Original reflectivity values
        gradient: Vertical gradient
        peak_height: Height of the BB peak
        heights: Array of height values

    Returns:
        Tuple of (bottom_height, top_height, amplitude_below, amplitude_above)
    """
    step = STEP
    peak_dbz = float(dbz.sel(height=peak_height).values)

    # Find bottom: look for maximum gradient change below peak
    bottom_height = -1 if peak_height == int(heights[0]) else None
    max_grad_diff_below = 0.0
    amplitude_below = 0.0 if bottom_height == -1 else None

    for k in range(peak_height, max(peak_height - 800, 0), -step):
        k_below = k - step
        k_below2 = k - 2 * step

        if k_below not in heights or (k_below2 not in heights and k_below2 >= 0):
            continue

        g1 = float(gradient.sel(height=k_below2).values) if k_below2 in heights else 0.0
        g2 = (
            float(gradient.sel(height=k_below).values) if k_below in heights else np.nan
        )

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
        g2 = (
            float(gradient.sel(height=k_above).values) if k_above in heights else np.nan
        )

        if np.isnan(g1) or np.isnan(g2):
            continue

        grad_diff = g1 - g2

        if grad_diff >= max_grad_diff_above:
            max_grad_diff_above = grad_diff
            top_height = k_above
            dbz_at_top = float(dbz.sel(height=k_above).values)
            amplitude_above = peak_dbz - dbz_at_top

    # The legacy comment specifies an 800-m fallback. Its loop-exit $k + 800
    # accidentally adds that distance twice; retain the stated physical limit.
    if top_height is None and peak_height + 800 in heights:
        top_height = peak_height + 800
        amplitude_above = peak_dbz - float(dbz.sel(height=top_height))

    return bottom_height, top_height, amplitude_below, amplitude_above


def _validate_bright_band(
    result: BrightBandResult,
    gradient: xr.DataArray,
    ds: xr.Dataset,
    lowest_height: int,
) -> BBRejectionReason | None:
    """Validate a bright band detection.

    Applies quality checks from the Perl implementation to reject
    false positives.

    Args:
        result: Preliminary BB detection result
        gradient: Vertical gradient for additional checks
        ds: Dataset with 'zcount' variable for sample count validation
        lowest_height: Lowest height level in the profile (for near-surface check)

    Returns:
        None if validation passes, or BBRejectionReason if rejected
    """
    # A peak at the lowest two levels with few or decreasing samples is
    # likely ground clutter. Check it first, also for bands failing the
    # checks below, so that the near-surface clutter correction follows.
    if result.peak_height in (lowest_height, lowest_height + STEP):
        if not _validate_near_surface_bb(ds, result.peak_height, result.top_height):
            return BBRejectionReason.NEAR_SURFACE_ZCOUNT

    if result.top_height is None or result.bottom_height is None:
        return BBRejectionReason.VALIDATION_FAILED

    # Outer checks use the original, ZCnt-qualified [3] gradient, unlike
    # candidate matching's [4] gradient gated by the corrected echo mask.
    outer_gradient = -compute_gradient(ds) * STEP if "zcount" in ds else gradient
    if result.top_height in outer_gradient["height"].values:
        grad_above = float(outer_gradient.sel(height=result.top_height).values)
        if not np.isfinite(grad_above):
            return BBRejectionReason.VALIDATION_FAILED
        if not np.isnan(grad_above) and result.amplitude_above is not None:
            # Convert downward-positive differences to Perl's upward sign.
            if -grad_above > result.amplitude_above / 2:
                return BBRejectionReason.VALIDATION_FAILED

    # For a 400-m band, reject steeply falling gradients on both outer sides.
    below = result.bottom_height - STEP
    if (
        result.thickness == 400
        and below in outer_gradient.height.values
        and result.top_height in outer_gradient.height.values
        and float(outer_gradient.sel(height=result.top_height)) > 2
        and float(outer_gradient.sel(height=below)) > 2
    ):
        return BBRejectionReason.VALIDATION_FAILED

    return None


def _validate_near_surface_bb(
    ds: xr.Dataset,
    bottom_height: int,
    top_height: int | None,
) -> bool:
    """Validate bright band when it appears at near-surface levels.

    When the BB peak is at the lowest 1-2 levels, additional checks
    on sample counts prevent false detection from ground clutter.

    Perl logic (lines 1051-1075 allprof_prodx2.pl):
        if ($bbalku == $alintaso or $bbalku == $alintaso + 200):
            if (Zcount[bbalku] < 500 or Zcount[zyla] < 500
                or Zcount[zyla]/Zcount[bbalku] < 0.7):
                bb = 0  # reject

    Args:
        ds: Dataset with 'zcount' variable
        bottom_height: Discrete BB peak height (historical argument name)
        top_height: BB top height

    Returns:
        True if the near-surface BB passes validation
    """
    if top_height is None:
        return False

    if "zcount" not in ds:
        # No sample count data, can't validate
        logger.debug("Near-surface BB validation skipped: no zcount data")
        return True

    heights = ds["height"].values
    zcount = ds["zcount"]

    # The retained argument name predates correcting bbalku to mean peak.
    if bottom_height not in heights or top_height not in heights:
        return False

    zcount_bottom = float(zcount.sel(height=bottom_height).values)
    zcount_top = float(zcount.sel(height=top_height).values)

    # Check minimum sample count at both levels
    if not np.isfinite(zcount_bottom) or zcount_bottom < NEAR_SURFACE_BB_MIN_ZCOUNT:
        logger.debug(
            "Near-surface BB rejected: Zcount at peak (%s) < %d",
            zcount_bottom,
            NEAR_SURFACE_BB_MIN_ZCOUNT,
        )
        return False

    if not np.isfinite(zcount_top) or zcount_top < NEAR_SURFACE_BB_MIN_ZCOUNT:
        logger.debug(
            "Near-surface BB rejected: Zcount at top (%s) < %d",
            zcount_top,
            NEAR_SURFACE_BB_MIN_ZCOUNT,
        )
        return False

    # Check ratio of sample counts
    if zcount_bottom > 0:
        ratio = zcount_top / zcount_bottom
        if ratio < NEAR_SURFACE_BB_MIN_ZCOUNT_RATIO:
            logger.debug(
                "Near-surface BB rejected: Zcount ratio (%.2f) < %.2f",
                ratio,
                NEAR_SURFACE_BB_MIN_ZCOUNT_RATIO,
            )
            return False

    logger.debug(
        "Near-surface BB validated: Zcount peak=%d, top=%d, ratio=%.2f",
        zcount_bottom,
        zcount_top,
        zcount_top / zcount_bottom if zcount_bottom > 0 else 0,
    )
    return True


def compute_dbz_gradient_per_step(ds: xr.Dataset) -> xr.DataArray:
    """Compute downward-positive raw reflectivity differences per step.

    Unlike the gradient in clutter.py (which is dBZ/m), this returns
    the difference between adjacent levels (dBZ per 200m step),
    matching the Perl BB detection logic.

    Args:
        ds: Dataset with raw 'lin_dbz' and working 'corrected_dbz'

    Returns:
        DataArray with gradient values, indexed at lower height of each pair
    """
    dbz = ds["lin_dbz"]

    # Compute: gradient[i] = dbz[i] - dbz[i+1] (positive = decreasing upward)
    # Using shift to get dbz at height+step
    dbz_shifted = dbz.shift(height=-1)
    gradient = dbz - dbz_shifted

    # Mask where either value is missing
    corrected = ds["corrected_dbz"]
    valid = (corrected > MDS) & (corrected.shift(height=-1) > MDS)
    gradient = gradient.where(valid)

    return gradient


def _refine_peak_height(ds: xr.Dataset, peak: int) -> float:
    """Reflectivity-weighted height; preserve the discrete restoration peak."""
    heights = ds.height.values
    values = ds["corrected_dbz"]
    if peak - 2 * STEP > 0:
        levels = [peak + offset for offset in BB_SPIKE_RESTORATION_OFFSETS]
        if not all(h in heights and float(values.sel(height=h)) > MDS for h in levels):
            return float(peak)
        if (
            float(values.sel(height=peak - 2 * STEP) - values.sel(height=peak - STEP))
            >= 2
            or float(
                values.sel(height=peak + 2 * STEP) - values.sel(height=peak + STEP)
            )
            >= 2
        ):
            return float(peak)
    elif peak - STEP > 0:
        levels = [peak - STEP, peak, peak + STEP]
        required = levels + [peak + 2 * STEP]
        if not all(
            h in heights and float(values.sel(height=h)) > MDS for h in required
        ):
            return float(peak)
    else:
        return float(peak)
    dbz = values.sel(height=levels).values
    weights = 10 ** ((dbz - np.max(dbz)) / 10)
    return float(np.average(levels, weights=weights))


def detect_bright_band(
    ds: xr.Dataset,
    layer_top: int | None = None,
    is_precipitation: bool = True,
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
        is_precipitation: Whether the lowest layer is a precipitation layer
            reaching the ground. A melting layer signature is not searched
            in other profiles, such as clear-air or altostratus echo.

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
    if not len(heights):
        return BrightBandResult(
            freezing_level_m=freezing_level,
            rejection_reason=BBRejectionReason.NO_CANDIDATES,
        )

    # Legacy: the search requires `$kerrostaulu[1][$j][6] == 1`.
    if not is_precipitation:
        return BrightBandResult(
            freezing_level_m=freezing_level,
            rejection_reason=BBRejectionReason.NOT_PRECIPITATION,
        )

    # Determine search bounds
    if layer_top is None:
        layer_top = int(heights[-1])

    # Skip detection if freezing level is not positive or layer too low
    if freezing_level is not None and freezing_level <= 0:
        return BrightBandResult(
            detected=False,
            freezing_level_m=freezing_level,
            rejection_reason=BBRejectionReason.FREEZING_LEVEL_INVALID,
        )

    if layer_top <= 1300:
        return BrightBandResult(
            detected=False,
            freezing_level_m=freezing_level,
            rejection_reason=BBRejectionReason.LAYER_TOO_LOW,
        )

    # Compute gradient (dBZ per step, not per meter)
    gradient = compute_dbz_gradient_per_step(ds)

    # Find candidate peaks
    candidates = _find_bb_candidates(dbz, gradient, heights, layer_top)

    if not candidates:
        return BrightBandResult(
            detected=False,
            freezing_level_m=freezing_level,
            rejection_reason=BBRejectionReason.NO_CANDIDATES,
        )

    # Select best candidate based on proximity to freezing level
    best_candidate = None
    min_distance = 500.0

    for h in candidates:
        if freezing_level is not None and freezing_level > 0:
            distance = abs(h - freezing_level)
            if distance <= 500 and distance <= min_distance:
                min_distance = abs(_refine_peak_height(ds, h) - freezing_level)
                best_candidate = h
        else:
            # Without freezing level, take highest candidate (most likely)
            if best_candidate is None or h > best_candidate:
                best_candidate = h

    if best_candidate is None:
        return BrightBandResult(
            detected=False,
            freezing_level_m=freezing_level,
            rejection_reason=BBRejectionReason.NO_CANDIDATES,
        )

    # Find boundaries
    bottom, top, amp_below, amp_above = _find_bb_boundaries(
        ds["lin_dbz"], gradient, best_candidate, heights
    )

    peak_dbz = float(ds["lin_dbz"].sel(height=best_candidate).values)

    result = BrightBandResult(
        detected=True,
        peak_height=best_candidate,
        bottom_height=bottom,
        top_height=top,
        peak_dbz=peak_dbz,
        amplitude_below=amp_below,
        amplitude_above=amp_above,
        freezing_level_m=freezing_level,
        refined_peak_height=_refine_peak_height(ds, best_candidate),
    )

    # Validate (pass dataset and lowest height for near-surface checks)
    lowest_height = int(heights[0])
    rejection_reason = _validate_bright_band(result, gradient, ds, lowest_height)
    if rejection_reason is not None:
        result.detected = False
        result.rejection_reason = rejection_reason

    return result


def apply_post_bb_clutter_correction(
    ds: xr.Dataset,
    bb_result: BrightBandResult,
) -> tuple[xr.Dataset, BrightBandResult]:
    """Apply ground clutter correction after BB detection/rejection.

    This function handles three scenarios from allprof_prodx2.pl (lines 1051-1144):

    1. Near-surface BB rejected due to zcount validation (lines 1051-1075):
       Apply clutter correction in lowest 600m above antenna.

    2. Valid BB with low bottom height ≤800m (lines 1104-1118):
       Apply clutter correction from BB bottom down to lowest level.

    3. No BB detected + low freezing level ≤1000m, OR valid BB with bottom >800m
       (lines 1121-1144): Apply clutter correction in lowest 600m with smoothing.

    The correction caps lower level values to upper level + 1 dB when the lower
    level exceeds this threshold, preventing artificial enhancement from clutter.
    Below an upper level without echo, the lower level becomes empty.

    Args:
        ds: Dataset with 'corrected_dbz' after initial processing
        bb_result: Result from detect_bright_band()

    Returns:
        Tuple of (corrected dataset, updated BrightBandResult with clutter flag)
    """
    heights = ds["height"].values
    if not len(heights):
        return ds, bb_result
    lowest = int(heights[0])
    freezing_level = bb_result.freezing_level_m
    scenario = _determine_clutter_scenario(bb_result, freezing_level)
    if scenario is None:
        return ds, bb_result
    scenarios = [scenario]
    if (
        scenario == "near_surface_rejected"
        and freezing_level is not None
        and freezing_level <= NO_BB_CLUTTER_FREEZING_LEVEL_M
    ):
        # These are separate Perl blocks, not mutually exclusive branches.
        scenarios.append("high_bb_or_no_bb")

    result = ds.copy(deep=True)
    corrected = result["corrected_dbz"].values.astype(float, copy=True)
    raw = result["lin_dbz"].values
    indices = {int(h): i for i, h in enumerate(heights)}
    # Levels absent from the VVP file are undefined in the legacy arrays. Only
    # the no-BB/high-BB block guards them (``$isotaulu[...][0] ne ""``),
    # taking its else branch; elsewhere the MDS fill is used as no echo.
    absent = set(result.attrs.get("missing_levels_m", ()))
    applied = False
    for scenario in scenarios:
        top = (
            bb_result.bottom_height
            if scenario == "low_bb_bottom"
            else lowest + POST_BB_CLUTTER_HEIGHT_M
        )
        for height in range(int(top), lowest, -STEP):
            if height not in indices or height - STEP not in indices:
                continue
            index, below = indices[height], indices[height - STEP]
            if not np.isfinite(raw[index]) or not np.isfinite(raw[below]):
                continue
            if scenario == "high_bb_or_no_bb" and (
                height in absent or height - STEP in absent
            ):
                # read_vvp() fills an absent level's raw value with MDS.
                corrected[below] = raw[below]
                continue
            threshold_reference = (
                corrected[index] if scenario == "high_bb_or_no_bb" else raw[index]
            )
            if raw[below] > threshold_reference + POST_BB_CLUTTER_THRESHOLD_DB:
                reference = (
                    raw[index]
                    if scenario == "near_surface_rejected"
                    else corrected[index]
                )
                corrected[below] = _cap_below(reference)
                applied = True
                if scenario == "high_bb_or_no_bb" and height + STEP in indices:
                    above = indices[height + STEP]
                    if corrected[above] - corrected[index] > LARGE_JUMP_THRESHOLD_DB:
                        corrected[index] = (corrected[index] + corrected[above]) / 2
                        corrected[below] = _cap_below(corrected[index])
            else:
                corrected[below] = raw[below]
    result["corrected_dbz"].values = corrected
    result.attrs["clutter_detected"] = bool(
        result.attrs.get("clutter_detected", False) or applied
    )
    return result, replace(
        bb_result,
        clutter_correction_applied=bb_result.clutter_correction_applied or applied,
    )


def _cap_below(reference: float) -> float:
    """Cap for the level below a reference level in the post-BB clutter steps.

    Perl caps at reference + 1 dB even when the reference has no echo, so an
    empty level turned the levels below into -44, -43, -42 dBZ "echo". An
    empty reference leaves the level below empty.
    """
    if reference <= MDS:
        return MDS
    return reference + POST_BB_CLUTTER_THRESHOLD_DB


def _determine_clutter_scenario(
    bb_result: BrightBandResult,
    freezing_level: float | None,
) -> str | None:
    """Determine which post-BB clutter correction scenario applies.

    Args:
        bb_result: BB detection result
        freezing_level: Freezing level in meters (above antenna)

    Returns:
        Scenario name or None if no correction needed
    """
    # Scenario A: Near-surface BB rejected due to zcount
    if bb_result.rejection_reason == BBRejectionReason.NEAR_SURFACE_ZCOUNT:
        return "near_surface_rejected"

    # Scenario B: Valid BB with low bottom height
    if bb_result.detected and bb_result.bottom_height is not None:
        if bb_result.bottom_height <= LOW_BB_HEIGHT_M:
            return "low_bb_bottom"

    # Scenario C: Valid BB with high bottom, OR no BB with low freezing level
    if bb_result.detected and bb_result.bottom_height is not None:
        if bb_result.bottom_height > LOW_BB_HEIGHT_M:
            return "high_bb_or_no_bb"

    if not bb_result.detected:
        if (
            freezing_level is not None
            and freezing_level <= NO_BB_CLUTTER_FREEZING_LEVEL_M
        ):
            return "high_bb_or_no_bb"

    return None


def restore_bb_spike(ds: xr.Dataset, bb_result: BrightBandResult) -> xr.Dataset:
    """Restore original dBZ values around bright band peak after spike smoothing.

    Spike smoothing may have smoothed the BB peak because it looks like a
    positive spike. However, the BB is a real physical enhancement, not noise.
    This function restores the original `lin_dbz` values at heights around the
    BB peak to preserve the true VPR shape for correction calculation.

    Based on allprof_prodx2.pl lines 1079-1101:
    "Korjataan mahdollisesti tasoitettu BB:n aiheuttama piikki
     takaisin alkuperäisiin arvoihin"
    (Restore the possibly smoothed spike caused by BB back to original values)

    Note: The separate cap_bb_peak_amplitude() function in vpr_correction.py
    handles the special case of surface BB with large amplitude (>10 dB).

    Args:
        ds: Dataset with 'corrected_dbz' and 'lin_dbz' variables
        bb_result: Result from detect_bright_band()

    Returns:
        Dataset with original values restored around BB peak
    """
    if not bb_result.detected or bb_result.peak_height is None:
        return ds

    peak = bb_result.peak_height
    heights = ds["height"].values
    lowest_height = int(heights[0])

    # Make a copy to avoid modifying original
    ds = ds.copy(deep=True)

    # Restore original lin_dbz values at heights around BB peak
    restored_count = 0
    for offset in BB_SPIKE_RESTORATION_OFFSETS:
        h = peak + offset
        if h in heights and h >= lowest_height:
            original_value = float(ds["lin_dbz"].sel(height=h).values)
            current_value = float(ds["corrected_dbz"].sel(height=h).values)

            if original_value != current_value:
                ds["corrected_dbz"].loc[{"height": h}] = original_value
                restored_count += 1
                logger.debug(
                    "Restored BB spike at %dm: %.1f -> %.1f dBZ",
                    h,
                    current_value,
                    original_value,
                )

    if restored_count > 0:
        logger.debug("Restored %d values around BB peak at %dm", restored_count, peak)

    return ds
