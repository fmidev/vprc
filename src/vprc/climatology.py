"""Climatological reflectivity profile generation.

This module generates a parametric vertical reflectivity profile based on
the freezing level height. This climatological profile serves as a fallback
when instantaneous VVP observations are unavailable or of poor quality.

Based on allprof_prodx2.pl section "klim.korjausprofiili" (climatological
correction profile), lines 1235-1300.

The climatological profile structure:
    - Base value at ground: 10 + (freezing_level_km * 10) dBZ
    - Below melting layer: constant at base value
    - Lower melting layer (FL-400 to FL-200): base + 3.5 dB
    - Melting layer peak (FL-200 to FL): base + 7.0 dB
    - Upper melting layer (FL to FL+200): base + 3.5 dB
    - Above FL+200: decays at -0.94 dB per 200m (if FL >= 0)
    - If FL < 0: decays at -0.66 dB per 200m from surface

Example for freezing level at 2000m:
    - Base: 10 + 2.0*10 = 30 dBZ
    - At 1600m (FL-400): 30 dBZ
    - At 1700m (FL-300): 33.5 dBZ (lower BB)
    - At 1900m (FL-100): 37.0 dBZ (BB peak)
    - At 2100m (FL+100): 33.5 dBZ (upper BB)
    - At 2300m (FL+300): 30 dBZ (above BB)
    - At 2500m: 29.06 dBZ (decay starts)
"""

import numpy as np
import xarray as xr

from .constants import MDS, STEP


def generate_climatological_profile(
    freezing_level_m: float,
    lowest_level_m: int = 100,
    max_height_m: int = 10000,
    step_m: int = STEP,
) -> xr.Dataset:
    """Generate climatological VPR profile based on freezing level.

    Creates a parametric profile representing typical stratiform precipitation
    with a melting layer (bright band) centered at the freezing level.

    Args:
        freezing_level_m: Freezing level height above antenna (m).
            Can be negative if freezing level is below radar.
        lowest_level_m: Lowest profile level above antenna (m).
            Should match the actual VVP profile grid (e.g., 100, 119).
        max_height_m: Maximum height for the profile (m).
        step_m: Vertical resolution, typically 200m to match VVP.

    Returns:
        xarray Dataset with:
        - 'clim_dbz': Climatological reflectivity profile (dBZ)
        - 'height': Coordinate in meters above antenna

    Example:
        >>> ds = generate_climatological_profile(freezing_level_m=2000)
        >>> ds['clim_dbz'].sel(height=1900)  # BB peak
        <xarray.DataArray 'clim_dbz' ()>
        array(37.)
    """
    # Round freezing level to nearest step for layer boundary alignment
    # Perl: $nollaraja_ps is used directly in comparisons
    fl = freezing_level_m

    # Base reflectivity: 10 + (FL_km * 10) dBZ
    # Perl: 10 + $nollaraja_ps / 1000 * 10
    base_dbz = 10 + fl / 1000 * 10

    # Create height array
    heights = np.arange(lowest_level_m, max_height_m, step_m)
    n_levels = len(heights)
    clim_dbz = np.full(n_levels, MDS, dtype=np.float64)

    # Build profile layer by layer
    for idx, h in enumerate(heights):
        if fl >= 0:
            # Freezing level at or above ground
            if h <= fl - 400:
                # Below melting layer: constant base
                clim_dbz[idx] = base_dbz
            elif h <= fl - 200:
                # Lower melting layer (FL-400 to FL-200): +3.5 dB
                clim_dbz[idx] = base_dbz + 3.5
            elif h <= fl:
                # Melting layer peak (FL-200 to FL): +7.0 dB
                clim_dbz[idx] = base_dbz + 7.0
            elif h <= fl + 200:
                # Upper melting layer (FL to FL+200): +3.5 dB
                clim_dbz[idx] = base_dbz + 3.5
            elif h <= fl + 400:
                # Just above melting layer (FL+200 to FL+400): back to base
                clim_dbz[idx] = base_dbz
            else:
                # Above FL+400: decay at -0.94 dB per 200m
                # Perl: $kortaulu[$i+200][$j][1] = $kortaulu[$i][$j][1] - 0.94
                levels_above = (h - (fl + 400)) // step_m
                clim_dbz[idx] = base_dbz - 0.94 * (levels_above + 1)
        else:
            # Freezing level below ground (winter conditions)
            # Base value at surface, decay upward at -0.66 dB per 200m
            # Perl: $kortaulu[$i+200][$j][1] = $kortaulu[$i][$j][1] - 0.66
            levels_above = (h - lowest_level_m) // step_m
            clim_dbz[idx] = base_dbz - 0.66 * levels_above

        # Clamp to MDS
        if clim_dbz[idx] < MDS:
            clim_dbz[idx] = MDS

    # Build xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "clim_dbz": (["height"], clim_dbz),
        },
        coords={
            "height": heights,
        },
        attrs={
            "freezing_level_m": freezing_level_m,
            "base_dbz": base_dbz,
            "description": "Climatological VPR profile based on freezing level",
        },
    )

    return ds


def get_clim_ground_reference(
    freezing_level_m: float,
    lowest_level_m: int = 100,
) -> float:
    """Get climatological ground reference value for VPR correction.

    When the melting layer is at or near the surface, the ground reference
    is the base value (without BB enhancement). Otherwise, it's the value
    at the lowest profile level.

    This implements the Perl logic for $korarvo_klim.

    Args:
        freezing_level_m: Freezing level height above antenna (m).
        lowest_level_m: Lowest profile level above antenna (m).

    Returns:
        Reference dBZ value for climatological correction calculation.
    """
    base_dbz = 10 + freezing_level_m / 1000 * 10
    fl = freezing_level_m

    # If freezing level is below ground or at lowest levels,
    # use base value without BB enhancement
    # Perl: if ($i == $alintaso or $i == $alintaso + 200) { $korarvo_klim = base }
    if fl < 0:
        # FL below ground: use base
        return base_dbz
    elif fl <= lowest_level_m + 400:
        # BB at or near ground: use base (avoid BB inflation at surface)
        return base_dbz
    else:
        # BB elevated: lowest level is at base value
        return base_dbz
