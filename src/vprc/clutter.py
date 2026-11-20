"""Ground clutter detection and removal based on allprof_prodx2.pl."""

import numpy as np
import xarray as xr

from .constants import (
    FREEZING_LEVEL_MIN,
    GROUND_CLUTTER_GRADIENT_THRESHOLD,
    MDS,
    MIN_SAMPLES,
    STEP,
)


def compute_gradient(ds: xr.Dataset, min_samples: int = MIN_SAMPLES) -> xr.DataArray:
    """Return upward-positive raw reflectivity gradients in dBZ/m.

    Both adjacent IRIS ZCnt values must exceed the threshold. IRIS Count
    describes wind samples and must not determine reflectivity validity.
    """
    dbz = ds["lin_dbz"]
    heights = ds["height"]
    gradient = dbz.diff("height", label="lower") / heights.diff("height", label="lower")
    gradient = gradient.reindex(height=heights)
    valid = ds["zcount"] > min_samples
    return gradient.where(valid & valid.shift(height=-1, fill_value=False))


def remove_ground_clutter(ds: xr.Dataset) -> xr.Dataset:
    """Correct the lowest three levels and retain the smoothing start height.

    ``clutter_detected`` records surviving clutter contamination.
    ``smoothing_start_height_m`` is the legacy ``alintaso + kortaso``.
    The input is not modified.

    Raises:
        ValueError: If the heights are not a contiguous ``STEP`` grid.
            ``read_vvp()`` fills levels missing from the file, matching the
            legacy height-indexed arrays.
    """
    result = ds.copy(deep=True)
    result["corrected_dbz"] = result["corrected_dbz"].astype(float)
    heights = result["height"].values
    if not len(heights):
        return result
    if np.any(np.diff(heights) != STEP):
        raise ValueError(
            f"Ground clutter removal requires a contiguous {STEP} m height grid"
        )
    lowest = int(heights[0])
    result.attrs["clutter_detected"] = False
    result.attrs["smoothing_start_height_m"] = lowest
    freezing_level = result.attrs.get("freezing_level_m")
    if freezing_level is not None and 0 < freezing_level <= FREEZING_LEVEL_MIN:
        return result

    gradient = compute_gradient(result)
    raw = result["lin_dbz"]
    working = result["corrected_dbz"]
    for offset in range(3):
        height = lowest + offset * STEP
        above = height + STEP
        if height not in heights or above not in heights:
            continue
        slope = float(gradient.sel(height=height))
        if (
            float(raw.sel(height=height)) > MDS
            and np.isfinite(slope)
            and slope < GROUND_CLUTTER_GRADIENT_THRESHOLD
        ):
            reference = float(raw.sel(height=above))
            for depth in range(offset + 1):
                target = lowest + depth * STEP
                extrapolated = (
                    reference
                    - (offset + 1 - depth) * GROUND_CLUTTER_GRADIENT_THRESHOLD * STEP
                )
                # Correct the legacy duplicated h1 assignment: extrapolate h0 too.
                working.loc[{"height": target}] = min(
                    extrapolated, float(raw.sel(height=target))
                )
            result.attrs["clutter_detected"] = True
            result.attrs["smoothing_start_height_m"] = above

    third = lowest + 2 * STEP
    if third in heights and float(raw.sel(height=third)) <= MDS:
        working.loc[{"height": lowest}] = MDS

    # Perl: "if there is no echo in levels 1-4, remove the clutter entirely".
    # The gradient [3] is $gmds when ZCnt is too low at either of its levels;
    # the levels up to such a pair are cleared. Two deviations:
    # - Perl ran the check after its level loop, with $i already at
    #   $alintaso + 200, so it checked levels 2-5 and cleared from level 2.
    #   A gap at level 5 then erased reliable levels 2-4 and left level 1.
    # - A single low-ZCnt level between reliable ones is a gap, not missing
    #   echo: only the gap is cleared, and spike smoothing fills it. Clearing
    #   the levels below it would detach precipitation from the ground.
    def reliable(height: int) -> bool:
        return height in heights and float(zcount.sel(height=height)) > MIN_SAMPLES

    zcount = result["zcount"]
    for offset in range(3):
        height = lowest + offset * STEP
        above = height + STEP
        if height not in heights or (reliable(height) and reliable(above)):
            continue
        gap = height if not reliable(height) else above
        if gap != lowest and reliable(gap - STEP) and reliable(gap + STEP):
            working.loc[{"height": gap}] = MDS
            continue
        working.loc[{"height": slice(lowest, height)}] = MDS
        result.attrs["clutter_detected"] = False
    return result
