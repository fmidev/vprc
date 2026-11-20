"""VPR (Vertical Profile Reflectivity) correction calculation.

This module computes range-dependent correction factors to compensate for
the vertical structure of reflectivity. The correction is computed by
convolving the observed vertical profile with the radar beam geometry.

Based on pystycappi.pl from the Koistinen & Pohjola algorithm.

Algorithm overview:
    1. Interpolate coarse VVP profile (200m) to fine grid (25m)
    2. For each range, compute beam center height using 4/3 Earth model
    3. Compute Gaussian beam weights for layers within beam volume
    4. Convolve (weighted average) profile with beam to get "observed" Z
    5. Correction = 10*log10(Z_ground / Z_observed)
"""

from dataclasses import dataclass

# Import for type annotation only
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from wradlib.georef import bin_altitude
from wradlib.trafo import idecibel

from .constants import (
    BB_PEAK_AMPLITUDE_CAP_DB,
    BB_SLOPE_HEIGHT_LIMIT_M,
    BB_SLOPE_THRESHOLD,
    BEAM_SUPPORT_WIDTHS,
    CLIMATOLOGICAL_WEIGHT,
    DEFAULT_BEAMWIDTH_DEG,
    DEFAULT_CAPPI_HEIGHTS_M,
    DEFAULT_HORIZON_ANGLE_DEG,
    DEFAULT_MAX_RANGE_KM,
    DEFAULT_RANGE_STEP_KM,
    EARTH_RADIUS_M,
    FINE_GRID_RESOLUTION_M,
    MAX_CORRECTION_DB,
    MAX_PROFILE_HEIGHT_M,
    MDS,
    MIN_CORRECTION_THRESHOLD_DB,
)

if TYPE_CHECKING:
    from .bright_band import BrightBandResult


# Quality weight calculation constants (from allprof_prodx2.pl lines 1188-1219)
# Minimum reflectivity sample count for a layer to contribute to quality.
MIN_ZCOUNT_FOR_QUALITY = 30
# Maximum height for quality weight integration (m above antenna)
MAX_QUALITY_HEIGHT_M = 5000
# Each of the lowest five levels must exceed this reflectivity sample count.
MIN_ZCOUNT_FOR_USABLE = 500


@dataclass
class VPRCorrectionResult:
    """Results of VPR correction calculation.

    Attributes:
        corrections: xarray Dataset with correction factors vs range.
            Contains 'cappi_correction_db', 'cappi_clim_correction_db',
            'cappi_blended_correction_db' and similar for elevations.
        z_ground_dbz: Reference reflectivity at ground level (dBZ)
        z_ground_clim_dbz: Climatological reference reflectivity (dBZ)
        quality_weight: Profile quality weight for blending (0-10 typical range)
        usable: True if corrections were successfully computed
    """

    corrections: xr.Dataset
    z_ground_dbz: float
    z_ground_clim_dbz: float | None = None
    quality_weight: float = 0.0
    usable: bool = True


def interpolate_to_fine_grid(
    ds: xr.Dataset,
    fine_resolution_m: int = FINE_GRID_RESOLUTION_M,
    max_height_m: int = MAX_PROFILE_HEIGHT_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate coarse VVP profile to fine vertical grid.

    Interpolation is performed in Z-space (linear reflectivity), not dBZ,
    to preserve physical meaning of reflectivity gradients.

    Profile levels keep their exact heights. pystycappi.pl stores each level
    in the 25 m bin ``int(h / 25)``, shifting the profile down by a
    radar-specific 1-21 m; this array-indexing artefact is not reproduced.
    Its effect on corrections is usually below 0.2 dB.

    Args:
        ds: Dataset with 'corrected_dbz' indexed by 'height'
        fine_resolution_m: Target vertical resolution (default 25m)
        max_height_m: Maximum height for output grid (default 15000m)

    Returns:
        Tuple of (fine_heights, fine_z) where:
        - fine_heights: 1D array of heights on fine grid (m)
        - fine_z: 1D array of linear reflectivity Z (mm^6/m^3)
    """
    coarse_heights = ds["height"].values
    coarse_dbz = ds["corrected_dbz"].values

    # A finite MDS floor preserves the limiting correction above echo tops.
    mds_z = idecibel(MDS)
    coarse_z = np.where(coarse_dbz > MDS, idecibel(coarse_dbz), mds_z)

    # Create fine grid
    n_layers = int(max_height_m / fine_resolution_m)
    fine_heights = np.arange(0, n_layers * fine_resolution_m, fine_resolution_m)
    echo_levels = np.flatnonzero(coarse_dbz > MDS + 1)
    if not len(echo_levels):
        return fine_heights, np.full(len(fine_heights), mds_z)
    top = echo_levels[-1] + 1

    # Interpolate in Z-space
    # Extrapolate below lowest level with constant value
    # Extrapolate above highest level with the MDS floor
    fine_z = np.interp(
        fine_heights,
        coarse_heights[:top],
        coarse_z[:top],
        left=coarse_z[0],  # Extend lowest value downward
        right=mds_z,
    )

    return fine_heights, fine_z


def compute_beam_height(
    range_m: float,
    elevation_deg: float,
    antenna_height_m: float,
) -> float:
    """Compute radar beam center height at given range.

    Uses wradlib's bin_altitude with the reference ellipsoid radius at 60 degrees
    and the 4/3 Earth model for standard atmospheric refraction.

    Args:
        range_m: Slant range from radar (meters)
        elevation_deg: Elevation angle (degrees)
        antenna_height_m: Antenna height above sea level (meters)

    Returns:
        Beam center height above sea level (meters)
    """
    return bin_altitude(
        range_m,
        elevation_deg,
        antenna_height_m,
        re=EARTH_RADIUS_M,
        ke=4.0 / 3.0,
    )


def solve_elevation_for_height(
    target_height_m: float,
    range_m: float,
    antenna_height_m: float,
    min_elevation_deg: float = 0.0,
) -> float:
    """Find elevation angle needed to reach target height at given range.

    Implements pseudo-CAPPI logic: if the required elevation is below
    the minimum available, return the minimum elevation.

    Args:
        target_height_m: Desired beam height (m above sea level)
        range_m: Slant range from radar (m)
        antenna_height_m: Antenna height above sea level (m)
        min_elevation_deg: Minimum available elevation angle (degrees)

    Returns:
        Elevation angle in degrees (may be capped at min_elevation_deg)
    """
    # Binary search for elevation angle
    low, high = -90.0, 90.0
    target = target_height_m

    for _ in range(50):  # Sufficient iterations for convergence
        mid = (low + high) / 2
        height = compute_beam_height(range_m, mid, antenna_height_m)
        if height < target:
            low = mid
        else:
            high = mid

    elevation = (low + high) / 2

    # Apply pseudo-CAPPI logic
    return max(elevation, min_elevation_deg)


def compute_beam_weight(
    angular_distance_deg: float,
    beamwidth_deg: float = DEFAULT_BEAMWIDTH_DEG,
) -> float:
    """Compute two-way Gaussian beam power weight.

    Based on Battan Eq. 9.4: W = 10^(-0.6 * (θ/θ_hp)^2)
    where θ is the angular distance from beam center and θ_hp is
    the one-way half-power beamwidth.

    Args:
        angular_distance_deg: Angular distance from beam center (degrees)
        beamwidth_deg: One-way half-power beamwidth (degrees)

    Returns:
        Power weight in range [0, 1]
    """
    return 10.0 ** (-0.6 * (angular_distance_deg / beamwidth_deg) ** 2)


def convolve_beam_with_profile(
    fine_heights: np.ndarray,
    fine_z: np.ndarray,
    beam_center_m: float,
    range_m: float,
    beamwidth_deg: float = DEFAULT_BEAMWIDTH_DEG,
    horizon_height_m: float = 0.0,
    *,
    beam_bounds_m: tuple[float, float] | None = None,
) -> float:
    """Convolve radar beam pattern with vertical Z profile.

    Computes the weighted average Z that the radar would observe,
    accounting for beam spreading and horizon blocking.

    Args:
        fine_heights: Height grid (m above antenna)
        fine_z: Linear reflectivity on fine grid
        beam_center_m: Beam center height (m above antenna)
        range_m: Slant range from radar (m)
        beamwidth_deg: One-way half-power beamwidth (degrees)
        horizon_height_m: Height of radar horizon (m), layers below are blocked
        beam_bounds_m: Lower and upper heights of the beam volume. When omitted,
            derive them from the beam center using the standard refraction model.

    Returns:
        Weighted average Z observed by beam (linear units)
    """
    if beamwidth_deg <= 0 or range_m <= 0:
        raise ValueError("Beamwidth and range must be positive")
    if len(fine_heights) < 2:
        raise ValueError("The fine grid must contain at least two heights")
    resolution = float(fine_heights[1] - fine_heights[0])
    if resolution <= 0:
        raise ValueError("The fine grid must be increasing")

    if beam_bounds_m is None:
        elevation = solve_elevation_for_height(
            beam_center_m, range_m, 0.0, min_elevation_deg=-90.0
        )
        beam_bounds_m = (
            compute_beam_height(
                range_m, elevation - BEAM_SUPPORT_WIDTHS * beamwidth_deg, 0.0
            ),
            compute_beam_height(
                range_m, elevation + BEAM_SUPPORT_WIDTHS * beamwidth_deg, 0.0
            ),
        )

    # Include blocked slices in the denominator, including those below antenna.
    # Match Perl's truncation and one extra slice below the beam's lower edge.
    lower_index = int(beam_bounds_m[0] / resolution) - 1
    upper_index = min(
        int(beam_bounds_m[1] / resolution), int(fine_heights[-1] / resolution)
    )
    indices = np.arange(lower_index, upper_index + 1)
    heights = indices * resolution
    height_diff = heights - beam_center_m
    angular_distance = np.rad2deg(np.arctan2(np.abs(height_diff), range_m))

    # Compute weights using Gaussian beam pattern
    weights = compute_beam_weight(angular_distance, beamwidth_deg)

    above_horizon = indices >= int(horizon_height_m / resolution)
    profile_z = np.interp(heights, fine_heights, fine_z, left=0.0)
    z_contribution = np.where(above_horizon, weights * profile_z, 0.0)
    weight_sum = np.sum(weights)

    if weight_sum == 0:
        return 0.0

    return np.sum(z_contribution) / weight_sum


def compute_correction_for_range(
    fine_heights: np.ndarray,
    fine_z: np.ndarray,
    z_ground: float,
    range_km: float,
    elevation_deg: float,
    antenna_height_m: float,
    beamwidth_deg: float = DEFAULT_BEAMWIDTH_DEG,
    max_correction_db: float = MAX_CORRECTION_DB,
    *,
    horizon_angle_deg: float = DEFAULT_HORIZON_ANGLE_DEG,
    previous_correction_db: float = 0.0,
    is_cappi: bool = False,
) -> tuple[float, float]:
    """Compute VPR correction at a single range.

    Args:
        fine_heights: Height grid (m above antenna)
        fine_z: Linear reflectivity on fine grid
        z_ground: Reference Z at ground level
        range_km: Range from radar (km)
        elevation_deg: Beam elevation angle (degrees)
        antenna_height_m: Antenna height (m)
        beamwidth_deg: Beamwidth (degrees)
        max_correction_db: Maximum allowed correction (dB)
        horizon_angle_deg: Angular terrain horizon (degrees).
        previous_correction_db: Correction to retain if the beam misses the
            entire integration grid (zero at the first range).
        is_cappi: Apply the legacy near-antenna CAPPI beam-height safeguard.

    Returns:
        Tuple of (correction_db, beam_height_m)
    """
    range_m = range_km * 1000.0

    # Compute beam center height (relative to antenna for profile lookup)
    beam_height_asl = compute_beam_height(range_m, elevation_deg, antenna_height_m)
    beam_height_antenna = beam_height_asl - antenna_height_m
    lower_angle = elevation_deg - BEAM_SUPPORT_WIDTHS * beamwidth_deg
    upper_angle = elevation_deg + BEAM_SUPPORT_WIDTHS * beamwidth_deg
    lower_height = (
        compute_beam_height(range_m, lower_angle, antenna_height_m) - antenna_height_m
    )
    upper_height = (
        compute_beam_height(range_m, upper_angle, antenna_height_m) - antenna_height_m
    )
    horizon_height = (
        compute_beam_height(
            range_m, max(horizon_angle_deg, lower_angle), antenna_height_m
        )
        - antenna_height_m
    )
    if is_cappi:
        resolution = float(fine_heights[1] - fine_heights[0])
        if beam_height_antenna < 2 * resolution:
            beam_height_antenna = 2 * resolution
            horizon_height = resolution
            lower_height = -4 * resolution
            upper_height = 8 * resolution

    # Convolve beam with profile
    z_observed = convolve_beam_with_profile(
        fine_heights,
        fine_z,
        beam_height_antenna,
        range_m,
        beamwidth_deg,
        horizon_height,
        beam_bounds_m=(lower_height, upper_height),
    )

    # Compute correction
    if z_ground <= 0:
        correction_db = 0.0
    elif z_observed <= 0:
        correction_db = previous_correction_db
    else:
        correction_db = 10.0 * np.log10(z_ground / z_observed)

    # Apply limits
    correction_db = np.clip(correction_db, -max_correction_db, max_correction_db)

    # Zero out negligible corrections
    if abs(correction_db) < MIN_CORRECTION_THRESHOLD_DB:
        correction_db = 0.0

    return correction_db, beam_height_antenna


def compute_quality_weight(
    ds: xr.Dataset,
    max_height_m: int = MAX_QUALITY_HEIGHT_M,
    *,
    is_precipitation: bool = True,
    layer_top_height_m: float | None = None,
) -> float:
    """Compute profile quality weight for correction blending.

    The quality weight indicates how reliable the instantaneous profile is
    for VPR correction. It measures reflectivity sample coverage, not echo
    strength: Perl's $isotaulu[][1] contains IRIS ZCnt.

    Based on allprof_prodx2.pl lines 1188-1219 ($wq calculation).

    Algorithm:
        1. Require precipitation and ZCnt > 500 at the lowest five levels
        2. Sum sample counts from ground to min(layer_top, 5km)
        3. Only include layers with ZCnt > 30
        4. Normalize by (5000 * num_valid_layers)

    Args:
        ds: Dataset with 'zcount' indexed by 'height'. Without sample counts,
            no observational quality can be established.
        max_height_m: Maximum integration height (default 5000m).
        is_precipitation: Whether the finalized profile is precipitation.
        layer_top_height_m: Top of the first precipitation layer. When omitted,
            infer the first contiguous echo layer from corrected_dbz.

    Returns:
        Quality weight (typically 0-10, where higher = better quality).
        Returns 0 if profile is not suitable for VPR correction.
    """
    if not is_precipitation or "zcount" not in ds:
        return 0.0

    heights = ds["height"].values
    counts = ds["zcount"].values
    if (
        len(counts) < 5
        or not np.all(np.isfinite(counts[:5]))
        or not np.all(counts[:5] > MIN_ZCOUNT_FOR_USABLE)
    ):
        return 0.0

    if layer_top_height_m is None:
        echo = ds["corrected_dbz"].values > MDS
        if not np.any(echo):
            return 0.0
        first = np.flatnonzero(echo)[0]
        gaps = np.flatnonzero(~echo[first:])
        last = first + gaps[0] - 1 if len(gaps) else len(heights) - 1
        layer_top_height_m = heights[last]

    mask = (
        (heights <= min(max_height_m, layer_top_height_m))
        & np.isfinite(counts)
        & (counts > MIN_ZCOUNT_FOR_QUALITY)
    )
    if not np.any(mask):
        return 0.0

    return float(np.mean(counts[mask]) / 5000)


def compute_bb_slope(
    dbz_values: np.ndarray,
    heights: np.ndarray,
    bb_bottom_height: int,
    bb_top_height: int,
) -> float:
    """Compute bright band slope in dBZ per 200m step.

    The slope indicates how steep the reflectivity gradient is within the
    bright band. Steep slopes near the surface can cause over-correction
    if not handled properly.

    Based on allprof_prodx2.pl line 1307:
        $bbkallistus = ($kortaulu[$zala][$j][0] - $kortaulu[$zyla][$j][0]) /
                       ($zyla - $zala) * 200

    Args:
        dbz_values: Array of dBZ values indexed by height
        heights: Array of heights (m above antenna)
        bb_bottom_height: Lower boundary of bright band (m)
        bb_top_height: Upper boundary of bright band (m)

    Returns:
        Slope in dBZ per 200m step. Positive values indicate decreasing
        reflectivity with height (typical BB shape).
    """
    # Find indices for BB boundaries
    bottom_idx = np.argmin(np.abs(heights - bb_bottom_height))
    top_idx = np.argmin(np.abs(heights - bb_top_height))

    dbz_bottom = dbz_values[bottom_idx]
    dbz_top = dbz_values[top_idx]

    height_diff = bb_top_height - bb_bottom_height
    if height_diff == 0:
        return 0.0

    # Slope in dBZ per 200m
    return (dbz_bottom - dbz_top) / height_diff * 200


def adjust_ground_reference(
    dbz_values: np.ndarray,
    heights: np.ndarray,
    bb_result: BrightBandResult,
    lowest_height: int,
    slope_threshold: float = BB_SLOPE_THRESHOLD,
    height_limit_m: int = BB_SLOPE_HEIGHT_LIMIT_M,
) -> tuple[float, bool]:
    """Adjust ground reference value when bright band is near surface.

    When a steep bright band is close to the surface, the enhanced reflectivity
    within the BB can cause over-correction. This function adjusts the ground
    reference to use an extrapolated value from above the BB instead.

    Based on allprof_prodx2.pl lines 1304-1318:
        - If BB peak is at lowest level: use value at BB top
        - If BB lower edge < 600m and slope > 0.2: extrapolate from BB top
        - Cap adjusted value at observed lowest-level dBZ

    Args:
        dbz_values: Array of dBZ values indexed by height
        heights: Array of heights (m above antenna)
        bb_result: BrightBandResult with BB boundaries
        lowest_height: Lowest profile level height (m)
        slope_threshold: Minimum slope to trigger adjustment (dBZ/200m)
        height_limit_m: Maximum BB bottom height for adjustment (m)

    Returns:
        Tuple of (adjusted_dbz, was_adjusted) where adjusted_dbz is the
        reference value to use and was_adjusted indicates if adjustment
        was applied.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Find lowest valid index
    valid_mask = dbz_values > MDS
    if not np.any(valid_mask):
        return MDS, False

    lowest_valid_idx = np.argmax(valid_mask)
    original_dbz = float(dbz_values[lowest_valid_idx])

    # No BB detected - use original value
    if not bb_result.detected:
        return original_dbz, False

    bb_bottom = bb_result.bottom_height
    bb_top = bb_result.top_height
    bb_peak = bb_result.peak_height

    if bb_bottom is None or bb_top is None or bb_peak is None:
        return original_dbz, False

    # Find dBZ at BB top
    top_idx = np.argmin(np.abs(heights - bb_top))
    dbz_at_top = float(dbz_values[top_idx])

    # Case 1: BB peak is at the lowest level
    # Perl: if ($bbhuippu == $alintaso) { $kortaulu[$alintaso][$j][0] = $isotaulu[$zyla][$j][5] }
    if bb_peak == lowest_height:
        logger.debug(
            "BB peak at lowest level (%dm), using BB top value: %.1f dBZ",
            lowest_height,
            dbz_at_top,
        )
        return dbz_at_top, True

    # Case 2: BB lower edge is above ground but near surface with steep slope
    # Perl: if ($zala > 0 and $zala < 600 and $bbkallistus > 0.2)
    if 0 < bb_bottom < height_limit_m:
        slope = compute_bb_slope(dbz_values, heights, bb_bottom, bb_top)

        if slope > slope_threshold:
            # Extrapolate from BB top downward
            # Perl: $kortaulu[$zyla][$j][0] + $zyla / 200 * 0.2
            adjusted_dbz = dbz_at_top + bb_top / 200 * 0.2

            # Safety cap: don't exceed observed lowest-level value
            # Perl: if ($isotaulu[$alintaso][$j][5] < $kortaulu[$alintaso][$j][0])
            if original_dbz < adjusted_dbz:
                logger.debug(
                    "BB slope adjustment capped: adjusted %.1f dBZ > observed %.1f dBZ",
                    adjusted_dbz,
                    original_dbz,
                )
                return original_dbz, False

            logger.debug(
                "BB slope adjustment applied: bottom=%dm, slope=%.2f dBZ/200m, "
                "original=%.1f dBZ -> adjusted=%.1f dBZ",
                bb_bottom,
                slope,
                original_dbz,
                adjusted_dbz,
            )
            return adjusted_dbz, True
        else:
            logger.debug(
                "BB near surface but slope %.2f <= threshold %.2f, no adjustment",
                slope,
                slope_threshold,
            )

    return original_dbz, False


def cap_bb_peak_amplitude(
    ds: xr.Dataset,
    bb_result: BrightBandResult,
    amplitude_cap_db: float = BB_PEAK_AMPLITUDE_CAP_DB,
) -> xr.Dataset:
    """Cap bright band peak amplitude when BB is at surface level.

    When a bright band sits at the lowest profile level with very high
    amplitude (difference from peak to top), this can cause excessive
    VPR correction. This function reduces the peak dBZ value to cap the
    effective amplitude.

    Original lin_dbz or bb_result.peak_dbz makes repeated application safe.
    Corrected-only callers without either must supply an uncapped profile.

    Based on allprof_prodx2.pl lines 1091-1098:
        if ( $ylaamp >= 10 and $bbalku == $alintaso ) {
            $isotaulu[$bbalku][$j][5] = $isotaulu[$bbalku][$j][0] - ( $ylaamp - 10 );
        }

    Args:
        ds: Dataset with 'corrected_dbz' indexed by 'height'
        bb_result: BrightBandResult with detection info
        amplitude_cap_db: Maximum allowed amplitude above peak (default 10 dB)

    Returns:
        Dataset with potentially modified corrected_dbz at BB peak.
        Returns original dataset if no capping needed.
    """
    import logging

    logger = logging.getLogger(__name__)

    if not bb_result.detected:
        return ds

    peak_height = bb_result.peak_height
    amplitude_above = bb_result.amplitude_above

    if peak_height is None or amplitude_above is None:
        return ds

    # Check if BB peak is at lowest level
    heights = ds["height"].values
    lowest_height = int(heights.min())

    if peak_height != lowest_height:
        return ds

    # Check if amplitude exceeds cap
    if amplitude_above <= amplitude_cap_db:
        return ds

    # Cap the peak amplitude by reducing the peak dBZ
    # Perl: $isotaulu[$bbalku][$j][5] = $isotaulu[$bbalku][$j][0] - ( $ylaamp - 10 )
    reduction = amplitude_above - amplitude_cap_db

    ds = ds.copy(deep=True)
    if "lin_dbz" in ds:
        original_peak_dbz = float(ds["lin_dbz"].sel(height=peak_height).values)
    elif bb_result.peak_dbz is not None:
        original_peak_dbz = float(bb_result.peak_dbz)
    else:
        original_peak_dbz = float(ds["corrected_dbz"].sel(height=peak_height).values)
    capped_peak_dbz = original_peak_dbz - reduction

    ds["corrected_dbz"].loc[dict(height=peak_height)] = capped_peak_dbz

    logger.debug(
        "BB peak amplitude capped at surface: amplitude_above=%.1f dBZ > cap=%.1f dBZ, "
        "peak reduced from %.1f to %.1f dBZ (reduction=%.1f dBZ)",
        amplitude_above,
        amplitude_cap_db,
        original_peak_dbz,
        capped_peak_dbz,
        reduction,
    )

    return ds


def compute_vpr_correction(
    ds: xr.Dataset,
    cappi_heights_m: tuple[int, ...] | None = None,
    elevation_angles_deg: tuple[float, ...] | None = None,
    max_range_km: int = DEFAULT_MAX_RANGE_KM,
    range_step_km: int = DEFAULT_RANGE_STEP_KM,
    min_elevation_deg: float = 0.5,
    beamwidth_deg: float | None = None,
    freezing_level_m: float | None = None,
    include_clim: bool = True,
    clim_weight: float = CLIMATOLOGICAL_WEIGHT,
    bb_result: BrightBandResult | None = None,
    is_precipitation: bool = True,
    layer_top_height_m: float | None = None,
    horizon_angle_deg: float | None = None,
) -> VPRCorrectionResult:
    """Compute VPR correction factors vs range for CAPPI heights and elevations.

    This is the main entry point for VPR correction calculation. Optionally
    computes climatological corrections and blends them with instantaneous
    corrections based on profile quality.

    Args:
        ds: Dataset with 'corrected_dbz' indexed by 'height'.
            Must have 'antenna_height_m' in attrs.
            Should have 'beamwidth_deg' in attrs (from radar config).
        cappi_heights_m: CAPPI heights above sea level to compute corrections for.
            Default: (500, 1000)
        elevation_angles_deg: Fixed elevation angles to compute corrections for.
            If None, uses angles from ds.attrs['elevation_angles'] (from VVP).
            Pass empty tuple () to disable elevation-based corrections.
        max_range_km: Maximum range for corrections (default 250 km)
        range_step_km: Range step size (default 1 km)
        min_elevation_deg: Minimum elevation angle for pseudo-CAPPI
        beamwidth_deg: Radar beamwidth (degrees). If None, reads from
            ds.attrs['beamwidth_deg'] or uses default.
        freezing_level_m: Freezing level for climatological profile (m above antenna).
            Required if include_clim=True. If None and include_clim=True,
            attempts to read from ds.attrs['freezing_level_m'].
        include_clim: Whether to compute climatological corrections (default True).
        clim_weight: Fixed weight for climatological correction in blending.
            Default 0.2 (from pystycappi_ka.pl).
        bb_result: Bright band detection result. If provided, enables slope
            adjustment when BB is near the surface with steep gradient.
        is_precipitation: Final precipitation classification for quality gating.
        layer_top_height_m: Top of the first precipitation layer for quality
            integration; inferred from the profile when omitted.
        horizon_angle_deg: Angular terrain horizon. When omitted, reads
            ds.attrs['horizon_angle_deg'], falling back to zero degrees.

    Returns:
        VPRCorrectionResult containing:
        - corrections: Dataset with dims (range_km, cappi_height) and optionally
          (range_km, elevation) for elevation-based corrections. Includes
          'cappi_correction_db' (instantaneous), 'cappi_clim_correction_db' (clim),
          and 'cappi_blended_correction_db' (weighted blend).
        - z_ground_dbz: Reference ground reflectivity
        - z_ground_clim_dbz: Climatological ground reference
        - quality_weight: Profile quality weight
        - usable: Whether corrections are valid

    Example:
        >>> ds = read_vvp("profile.txt")
        >>> ds = remove_ground_clutter(ds)
        >>> ds = smooth_spikes(ds)
        >>> result = compute_vpr_correction(ds, freezing_level_m=2000)
        >>> result.corrections['cappi_correction_db'].sel(cappi_height=500, range_km=100)
        >>> result.corrections['cappi_blended_correction_db'].sel(cappi_height=500, range_km=100)
    """
    import logging

    from .climatology import generate_climatological_profile, get_clim_ground_reference

    logger = logging.getLogger(__name__)

    if cappi_heights_m is None:
        cappi_heights_m = DEFAULT_CAPPI_HEIGHTS_M

    # Get elevation angles from parameter or dataset attrs
    if elevation_angles_deg is None:
        elevation_angles_deg = tuple(ds.attrs.get("elevation_angles", []))

    # Get beamwidth from parameter, dataset attrs, or use default
    if beamwidth_deg is None:
        beamwidth_deg = ds.attrs.get("beamwidth_deg", DEFAULT_BEAMWIDTH_DEG)
    if horizon_angle_deg is None:
        horizon_angle_deg = ds.attrs.get("horizon_angle_deg", DEFAULT_HORIZON_ANGLE_DEG)

    antenna_height_m = ds.attrs.get("antenna_height_m", 0)

    # Get freezing level for climatological profile
    if freezing_level_m is None:
        freezing_level_m = ds.attrs.get("freezing_level_m")

    # Apply BB peak amplitude cap if BB is at surface with high amplitude
    # This modifies the profile before interpolation/correction calculation
    if bb_result is not None:
        ds = cap_bb_peak_amplitude(ds, bb_result)

    # Compute quality weight
    quality_weight = compute_quality_weight(
        ds,
        is_precipitation=is_precipitation,
        layer_top_height_m=layer_top_height_m,
    )

    # Interpolate profile to fine grid
    fine_heights, fine_z = interpolate_to_fine_grid(ds)

    # Determine ground reference Z (lowest valid level)
    heights = ds["height"].values
    dbz_values = ds["corrected_dbz"].values
    valid_mask = dbz_values > MDS

    if not np.any(valid_mask) or not is_precipitation:
        # Non-precipitation profiles cannot supply observed corrections.
        return _create_climatology_only_result(
            ds=ds,
            cappi_heights_m=cappi_heights_m,
            elevation_angles_deg=elevation_angles_deg,
            max_range_km=max_range_km,
            range_step_km=range_step_km,
            min_elevation_deg=min_elevation_deg,
            beamwidth_deg=beamwidth_deg,
            freezing_level_m=freezing_level_m if include_clim else None,
            clim_weight=clim_weight,
            horizon_angle_deg=horizon_angle_deg,
        )

    # Apply bright band slope adjustment if BB result provided
    lowest_height = int(heights[0])
    if bb_result is not None:
        z_ground_dbz, adjusted = adjust_ground_reference(
            dbz_values, heights, bb_result, lowest_height
        )
        if adjusted:
            logger.info(
                "Ground reference adjusted for BB slope: %.1f dBZ", z_ground_dbz
            )
    else:
        lowest_valid_idx = np.argmax(valid_mask)
        z_ground_dbz = float(dbz_values[lowest_valid_idx])

    z_ground = idecibel(z_ground_dbz)

    # Create range array
    ranges_km = np.arange(range_step_km, max_range_km + range_step_km, range_step_km)
    n_ranges = len(ranges_km)
    n_heights = len(cappi_heights_m)
    n_elevs = len(elevation_angles_deg)

    # Allocate output arrays for CAPPI heights
    corrections = np.zeros((n_ranges, n_heights))
    beam_heights = np.zeros((n_ranges, n_heights))

    # Compute corrections for each CAPPI height and range
    for j, cappi_height in enumerate(cappi_heights_m):
        for i, range_km in enumerate(ranges_km):
            range_m = range_km * 1000.0

            # Find elevation angle for this CAPPI height
            elevation = solve_elevation_for_height(
                cappi_height,
                range_m,
                antenna_height_m,
                min_elevation_deg,
            )

            # Compute correction
            corr, beam_h = compute_correction_for_range(
                fine_heights,
                fine_z,
                z_ground,
                range_km,
                elevation,
                antenna_height_m,
                beamwidth_deg,
                horizon_angle_deg=horizon_angle_deg,
                previous_correction_db=corrections[i - 1, j] if i else 0.0,
                is_cappi=True,
            )

            corrections[i, j] = corr
            beam_heights[i, j] = beam_h

    # Compute climatological corrections if requested and freezing level available
    clim_corrections = None
    z_ground_clim_dbz = None
    z_ground_clim = None

    if include_clim and freezing_level_m is not None:
        # Generate climatological profile
        lowest_level_m = int(heights[0])
        clim_ds = generate_climatological_profile(
            freezing_level_m=freezing_level_m,
            lowest_level_m=lowest_level_m,
        )

        # Create a dataset compatible with interpolate_to_fine_grid
        clim_profile_ds = xr.Dataset(
            data_vars={"corrected_dbz": (["height"], clim_ds["clim_dbz"].values)},
            coords={"height": clim_ds["height"].values},
        )

        # Interpolate clim profile to fine grid
        clim_fine_heights, clim_fine_z = interpolate_to_fine_grid(clim_profile_ds)

        # Get climatological ground reference
        z_ground_clim_dbz = get_clim_ground_reference(freezing_level_m, lowest_level_m)
        z_ground_clim = idecibel(z_ground_clim_dbz)

        # Allocate clim correction arrays
        clim_corrections = np.zeros((n_ranges, n_heights))

        # Compute climatological corrections for each CAPPI height and range
        for j, cappi_height in enumerate(cappi_heights_m):
            for i, range_km in enumerate(ranges_km):
                range_m = range_km * 1000.0

                elevation = solve_elevation_for_height(
                    cappi_height,
                    range_m,
                    antenna_height_m,
                    min_elevation_deg,
                )

                corr, _ = compute_correction_for_range(
                    clim_fine_heights,
                    clim_fine_z,
                    z_ground_clim,
                    range_km,
                    elevation,
                    antenna_height_m,
                    beamwidth_deg,
                    horizon_angle_deg=horizon_angle_deg,
                    previous_correction_db=clim_corrections[i - 1, j] if i else 0.0,
                    is_cappi=True,
                )

                clim_corrections[i, j] = corr

    # Build output Dataset with CAPPI corrections
    data_vars = {
        "cappi_correction_db": (["range_km", "cappi_height"], corrections),
        "cappi_beam_height_m": (["range_km", "cappi_height"], beam_heights),
    }

    # Add climatological and blended corrections if computed
    if clim_corrections is not None:
        data_vars["cappi_clim_correction_db"] = (
            ["range_km", "cappi_height"],
            clim_corrections,
        )

        # Compute blended correction
        # Perl: (clim_weight * corr_clim + quality_weight * corr_sade) / (clim_weight + quality_weight)
        if quality_weight > 0:
            blended = (
                clim_weight * clim_corrections + quality_weight * corrections
            ) / (clim_weight + quality_weight)
        else:
            # No quality weight: use climatological only
            blended = clim_corrections.copy()

        data_vars["cappi_blended_correction_db"] = (
            ["range_km", "cappi_height"],
            blended,
        )

    coords = {
        "range_km": ranges_km,
        "cappi_height": list(cappi_heights_m),
    }

    # Compute corrections for fixed elevation angles if provided
    if n_elevs > 0:
        elev_corrections = np.zeros((n_ranges, n_elevs))
        elev_beam_heights = np.zeros((n_ranges, n_elevs))

        for j, elev_deg in enumerate(elevation_angles_deg):
            for i, range_km in enumerate(ranges_km):
                # Compute correction for fixed elevation
                corr, beam_h = compute_correction_for_range(
                    fine_heights,
                    fine_z,
                    z_ground,
                    range_km,
                    elev_deg,
                    antenna_height_m,
                    beamwidth_deg,
                    horizon_angle_deg=horizon_angle_deg,
                    previous_correction_db=elev_corrections[i - 1, j] if i else 0.0,
                )

                elev_corrections[i, j] = corr
                elev_beam_heights[i, j] = beam_h

        # Add elevation-based corrections to dataset
        data_vars["elev_correction_db"] = (["range_km", "elevation"], elev_corrections)
        data_vars["elev_beam_height_m"] = (["range_km", "elevation"], elev_beam_heights)
        coords["elevation"] = list(elevation_angles_deg)

        # Add climatological elevation corrections if computed
        if include_clim and freezing_level_m is not None:
            elev_clim_corrections = np.zeros((n_ranges, n_elevs))

            for j, elev_deg in enumerate(elevation_angles_deg):
                for i, range_km in enumerate(ranges_km):
                    corr, _ = compute_correction_for_range(
                        clim_fine_heights,
                        clim_fine_z,
                        z_ground_clim,
                        range_km,
                        elev_deg,
                        antenna_height_m,
                        beamwidth_deg,
                        horizon_angle_deg=horizon_angle_deg,
                        previous_correction_db=(
                            elev_clim_corrections[i - 1, j] if i else 0.0
                        ),
                    )
                    elev_clim_corrections[i, j] = corr

            data_vars["elev_clim_correction_db"] = (
                ["range_km", "elevation"],
                elev_clim_corrections,
            )

            # Compute blended elevation corrections
            if quality_weight > 0:
                elev_blended = (
                    clim_weight * elev_clim_corrections
                    + quality_weight * elev_corrections
                ) / (clim_weight + quality_weight)
            else:
                elev_blended = elev_clim_corrections.copy()

            data_vars["elev_blended_correction_db"] = (
                ["range_km", "elevation"],
                elev_blended,
            )

    # Determine effective quality weight for compositing
    # If observed profile has zero quality but climatology is available,
    # use clim_weight as effective quality (climatology-only blending)
    effective_quality_weight = quality_weight
    is_climatology_only = False
    if quality_weight <= 0 and clim_corrections is not None:
        effective_quality_weight = clim_weight
        is_climatology_only = True

    correction_ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "radar": ds.attrs.get("radar", "unknown"),
            "timestamp": str(ds.attrs.get("timestamp", "")),
            "z_ground_dbz": z_ground_dbz,
            "z_ground_clim_dbz": z_ground_clim_dbz,
            "quality_weight": quality_weight,
            "clim_weight": clim_weight if include_clim else None,
            "freezing_level_m": freezing_level_m,
            "antenna_height_m": antenna_height_m,
            "beamwidth_deg": beamwidth_deg,
            "min_elevation_deg": min_elevation_deg,
            "horizon_angle_deg": horizon_angle_deg,
            "climatology_only": is_climatology_only,
        },
    )

    return VPRCorrectionResult(
        corrections=correction_ds,
        z_ground_dbz=z_ground_dbz,
        z_ground_clim_dbz=z_ground_clim_dbz,
        quality_weight=effective_quality_weight,
        usable=True,
    )


def _create_climatology_only_result(
    ds: xr.Dataset,
    cappi_heights_m: tuple[int, ...],
    elevation_angles_deg: tuple[float, ...],
    max_range_km: int,
    range_step_km: int,
    min_elevation_deg: float,
    beamwidth_deg: float,
    freezing_level_m: float | None,
    clim_weight: float,
    horizon_angle_deg: float = DEFAULT_HORIZON_ANGLE_DEG,
) -> VPRCorrectionResult:
    """Create climatology-only VPR correction result.

    Used when no valid echo exists but freezing level is available.
    Returns corrections based purely on the climatological profile.

    Args:
        ds: Dataset with profile metadata (antenna_height_m, etc.)
        cappi_heights_m: CAPPI heights to compute corrections for
        elevation_angles_deg: Fixed elevation angles
        max_range_km: Maximum range for corrections
        range_step_km: Range step size
        min_elevation_deg: Minimum elevation angle for pseudo-CAPPI
        beamwidth_deg: Radar beamwidth
        freezing_level_m: Freezing level for climatological profile
        clim_weight: Climatology weight for quality_weight in result
        horizon_angle_deg: Angular terrain horizon.

    Returns:
        VPRCorrectionResult with climatology-only corrections
    """
    from .climatology import generate_climatological_profile, get_clim_ground_reference

    ranges_km = np.arange(range_step_km, max_range_km + range_step_km, range_step_km)
    n_ranges = len(ranges_km)
    n_heights = len(cappi_heights_m)
    n_elevs = len(elevation_angles_deg)

    antenna_height_m = ds.attrs.get("antenna_height_m", 0)
    heights = ds["height"].values
    lowest_level_m = int(heights[0]) if len(heights) > 0 else 100

    # If no freezing level, return empty result with zeros
    if freezing_level_m is None:
        data_vars = {
            "cappi_correction_db": (
                ["range_km", "cappi_height"],
                np.zeros((n_ranges, n_heights)),
            ),
            "cappi_beam_height_m": (
                ["range_km", "cappi_height"],
                np.zeros((n_ranges, n_heights)),
            ),
        }
        coords = {
            "range_km": ranges_km,
            "cappi_height": list(cappi_heights_m),
        }
        if n_elevs > 0:
            data_vars["elev_correction_db"] = (
                ["range_km", "elevation"],
                np.zeros((n_ranges, n_elevs)),
            )
            data_vars["elev_beam_height_m"] = (
                ["range_km", "elevation"],
                np.zeros((n_ranges, n_elevs)),
            )
            coords["elevation"] = list(elevation_angles_deg)

        return VPRCorrectionResult(
            corrections=xr.Dataset(data_vars=data_vars, coords=coords),
            z_ground_dbz=MDS,
            usable=False,
        )

    # Generate climatological profile
    clim_ds = generate_climatological_profile(
        freezing_level_m=freezing_level_m,
        lowest_level_m=lowest_level_m,
    )

    # Create dataset compatible with interpolate_to_fine_grid
    clim_profile_ds = xr.Dataset(
        data_vars={"corrected_dbz": (["height"], clim_ds["clim_dbz"].values)},
        coords={"height": clim_ds["height"].values},
    )

    # Interpolate clim profile to fine grid
    clim_fine_heights, clim_fine_z = interpolate_to_fine_grid(clim_profile_ds)

    # Get climatological ground reference
    z_ground_clim_dbz = get_clim_ground_reference(freezing_level_m, lowest_level_m)
    z_ground_clim = idecibel(z_ground_clim_dbz)

    # Compute climatological CAPPI corrections
    clim_corrections = np.zeros((n_ranges, n_heights))
    beam_heights = np.zeros((n_ranges, n_heights))

    for j, cappi_height in enumerate(cappi_heights_m):
        for i, range_km in enumerate(ranges_km):
            range_m = range_km * 1000.0

            elevation = solve_elevation_for_height(
                cappi_height,
                range_m,
                antenna_height_m,
                min_elevation_deg,
            )

            corr, beam_h = compute_correction_for_range(
                clim_fine_heights,
                clim_fine_z,
                z_ground_clim,
                range_km,
                elevation,
                antenna_height_m,
                beamwidth_deg,
                horizon_angle_deg=horizon_angle_deg,
                previous_correction_db=clim_corrections[i - 1, j] if i else 0.0,
                is_cappi=True,
            )

            clim_corrections[i, j] = corr
            beam_heights[i, j] = beam_h

    # Build output Dataset - use clim corrections as both instant and blended
    data_vars = {
        "cappi_correction_db": (["range_km", "cappi_height"], clim_corrections),
        "cappi_beam_height_m": (["range_km", "cappi_height"], beam_heights),
        "cappi_clim_correction_db": (["range_km", "cappi_height"], clim_corrections),
        "cappi_blended_correction_db": (["range_km", "cappi_height"], clim_corrections),
    }
    coords = {
        "range_km": ranges_km,
        "cappi_height": list(cappi_heights_m),
    }

    # Compute elevation corrections if requested
    if n_elevs > 0:
        elev_corrections = np.zeros((n_ranges, n_elevs))
        elev_beam_heights = np.zeros((n_ranges, n_elevs))

        for j, elev_deg in enumerate(elevation_angles_deg):
            for i, range_km in enumerate(ranges_km):
                corr, beam_h = compute_correction_for_range(
                    clim_fine_heights,
                    clim_fine_z,
                    z_ground_clim,
                    range_km,
                    elev_deg,
                    antenna_height_m,
                    beamwidth_deg,
                    horizon_angle_deg=horizon_angle_deg,
                    previous_correction_db=elev_corrections[i - 1, j] if i else 0.0,
                )
                elev_corrections[i, j] = corr
                elev_beam_heights[i, j] = beam_h

        data_vars["elev_correction_db"] = (["range_km", "elevation"], elev_corrections)
        data_vars["elev_beam_height_m"] = (["range_km", "elevation"], elev_beam_heights)
        data_vars["elev_clim_correction_db"] = (
            ["range_km", "elevation"],
            elev_corrections,
        )
        data_vars["elev_blended_correction_db"] = (
            ["range_km", "elevation"],
            elev_corrections,
        )
        coords["elevation"] = list(elevation_angles_deg)

    correction_ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "radar": ds.attrs.get("radar", "unknown"),
            "timestamp": str(ds.attrs.get("timestamp", "")),
            "z_ground_dbz": z_ground_clim_dbz,
            "z_ground_clim_dbz": z_ground_clim_dbz,
            "quality_weight": 0.0,  # No observed profile quality
            "clim_weight": clim_weight,
            "freezing_level_m": freezing_level_m,
            "antenna_height_m": antenna_height_m,
            "beamwidth_deg": beamwidth_deg,
            "horizon_angle_deg": horizon_angle_deg,
            "climatology_only": True,
        },
    )

    return VPRCorrectionResult(
        corrections=correction_ds,
        z_ground_dbz=z_ground_clim_dbz,
        z_ground_clim_dbz=z_ground_clim_dbz,
        quality_weight=clim_weight,  # Use clim_weight as effective quality for compositing
        usable=True,
    )
