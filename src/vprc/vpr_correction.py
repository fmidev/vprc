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

import numpy as np
import xarray as xr
from wradlib.georef import bin_altitude
from wradlib.trafo import decibel, idecibel

from .constants import (
    MDS,
    STEP,
    FINE_GRID_RESOLUTION_M,
    MAX_PROFILE_HEIGHT_M,
    DEFAULT_BEAMWIDTH_DEG,
    MAX_CORRECTION_DB,
    MIN_CORRECTION_THRESHOLD_DB,
    DEFAULT_CAPPI_HEIGHTS_M,
    DEFAULT_MAX_RANGE_KM,
    DEFAULT_RANGE_STEP_KM,
)


# Quality weight calculation constants (from allprof_prodx2.pl lines 1188-1219)
# Minimum linear Z for layer to contribute to quality weight (~15 dBZ)
MIN_Z_FOR_QUALITY = 30  # mm^6/m^3
# Maximum height for quality weight integration (m above antenna)
MAX_QUALITY_HEIGHT_M = 5000
# Minimum linear Z at lowest 5 levels for profile to be usable (~27 dBZ)
MIN_Z_FOR_USABLE = 500  # mm^6/m^3


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

    # Convert dBZ to linear Z for interpolation
    # Handle MDS values by setting them to very small Z
    coarse_z = np.where(coarse_dbz > MDS, idecibel(coarse_dbz), 0.0)

    # Create fine grid
    n_layers = int(max_height_m / fine_resolution_m)
    fine_heights = np.arange(0, n_layers * fine_resolution_m, fine_resolution_m)

    # Interpolate in Z-space
    # Extrapolate below lowest level with constant value
    # Extrapolate above highest level with zero (no echo)
    fine_z = np.interp(
        fine_heights,
        coarse_heights,
        coarse_z,
        left=coarse_z[0],  # Extend lowest value downward
        right=0.0,  # No echo above profile top
    )

    return fine_heights, fine_z


def compute_beam_height(
    range_m: float,
    elevation_deg: float,
    antenna_height_m: float,
) -> float:
    """Compute radar beam center height at given range.

    Uses wradlib's bin_altitude with 4/3 Earth radius model for
    standard atmospheric refraction.

    Args:
        range_m: Slant range from radar (meters)
        elevation_deg: Elevation angle (degrees)
        antenna_height_m: Antenna height above sea level (meters)

    Returns:
        Beam center height above sea level (meters)
    """
    return bin_altitude(range_m, elevation_deg, antenna_height_m, ke=4.0 / 3.0)


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
    low, high = -2.0, 45.0
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

    Returns:
        Weighted average Z observed by beam (linear units)
    """
    # Compute angular distance from beam center for each layer
    # θ = atan2(height_diff, range) in degrees
    height_diff = fine_heights - beam_center_m
    angular_distance = np.rad2deg(np.arctan2(np.abs(height_diff), range_m))

    # Compute weights using Gaussian beam pattern
    weights = compute_beam_weight(angular_distance, beamwidth_deg)

    # Limit to layers within ~3 beamwidths (negligible contribution beyond)
    max_angle = 3.0 * beamwidth_deg
    in_beam = angular_distance <= max_angle

    # Apply horizon blocking: layers below horizon contribute zero Z
    # but their weights still count (reduces observed Z)
    above_horizon = fine_heights >= horizon_height_m

    # Weighted sum of Z (only above horizon contributes Z)
    z_contribution = np.where(above_horizon & in_beam, weights * fine_z, 0.0)
    weight_sum = np.sum(np.where(in_beam, weights, 0.0))

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

    Returns:
        Tuple of (correction_db, beam_height_m)
    """
    range_m = range_km * 1000.0

    # Compute beam center height (relative to antenna for profile lookup)
    beam_height_asl = compute_beam_height(range_m, elevation_deg, antenna_height_m)
    beam_height_antenna = beam_height_asl - antenna_height_m

    # Convolve beam with profile
    z_observed = convolve_beam_with_profile(
        fine_heights, fine_z, beam_height_antenna, range_m, beamwidth_deg
    )

    # Compute correction
    if z_observed <= 0 or z_ground <= 0:
        correction_db = 0.0
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
) -> float:
    """Compute profile quality weight for correction blending.

    The quality weight indicates how reliable the instantaneous profile is
    for VPR correction. Higher values indicate stronger, more continuous
    precipitation echo suitable for VPR correction.

    Based on allprof_prodx2.pl lines 1188-1219 ($wq calculation).

    Algorithm:
        1. Check that lowest 5 levels have Z > 500 mm^6/m^3 (~27 dBZ)
        2. Sum linear Z values from ground to min(layer_top, 5km)
        3. Only include layers with Z > 30 mm^6/m^3 (~15 dBZ)
        4. Normalize by (5000 * num_valid_layers)

    Args:
        ds: Dataset with 'corrected_dbz' indexed by 'height'.
        max_height_m: Maximum integration height (default 5000m).

    Returns:
        Quality weight (typically 0-10, where higher = better quality).
        Returns 0 if profile is not suitable for VPR correction.
    """
    heights = ds["height"].values
    dbz_values = ds["corrected_dbz"].values

    # Convert to linear Z
    z_values = np.where(dbz_values > MDS, idecibel(dbz_values), 0.0)

    # Check lowest 5 levels for sufficient echo strength
    # Perl: $isotaulu[$alintaso][$j][1] > 500 and ... (5 levels)
    n_required = min(5, len(z_values))
    if not np.all(z_values[:n_required] >= MIN_Z_FOR_USABLE):
        return 0.0

    # Integrate Z up to max_height_m, excluding weak layers
    mask = (heights <= max_height_m) & (z_values >= MIN_Z_FOR_QUALITY)

    if not np.any(mask):
        return 0.0

    # Sum Z values where mask is True (σ=1 for strong layers)
    # Perl: $wq_yla = $wq_yla + $isotaulu[$i][$j][1] * $sigma
    z_sum = np.sum(z_values[mask])

    # Normalize by 5000 * number of valid layers
    # Perl: $wq_ala = $wq_ala + 5000 * $sigma
    n_valid = np.sum(mask)
    normalization = 5000 * n_valid

    if normalization == 0:
        return 0.0

    return z_sum / normalization


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
    clim_weight: float = 0.2,
) -> VPRCorrectionResult:
    """Compute VPR correction factors vs range for CAPPI heights and elevations.

    This is the main entry point for VPR correction calculation. Optionally
    computes climatological corrections and blends them with instantaneous
    corrections based on profile quality.

    Args:
        ds: Dataset with 'corrected_dbz' indexed by 'height'.
            Must have 'antenna_height_m' in attrs.
            Should have 'beamwidth_deg' in attrs (from radar config).
        cappi_heights_m: CAPPI heights to compute corrections for.
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
    from .climatology import generate_climatological_profile, get_clim_ground_reference

    if cappi_heights_m is None:
        cappi_heights_m = DEFAULT_CAPPI_HEIGHTS_M

    # Get elevation angles from parameter or dataset attrs
    if elevation_angles_deg is None:
        elevation_angles_deg = tuple(ds.attrs.get("elevation_angles", []))

    # Get beamwidth from parameter, dataset attrs, or use default
    if beamwidth_deg is None:
        beamwidth_deg = ds.attrs.get("beamwidth_deg", DEFAULT_BEAMWIDTH_DEG)

    antenna_height_m = ds.attrs.get("antenna_height_m", 0)

    # Get freezing level for climatological profile
    if freezing_level_m is None:
        freezing_level_m = ds.attrs.get("freezing_level_m")

    # Compute quality weight
    quality_weight = compute_quality_weight(ds)

    # Interpolate profile to fine grid
    fine_heights, fine_z = interpolate_to_fine_grid(ds)

    # Determine ground reference Z (lowest valid level)
    heights = ds["height"].values
    dbz_values = ds["corrected_dbz"].values
    valid_mask = dbz_values > MDS

    if not np.any(valid_mask):
        # No valid echo - return climatology-only result if freezing level available
        return _create_climatology_only_result(
            ds=ds,
            cappi_heights_m=cappi_heights_m,
            elevation_angles_deg=elevation_angles_deg,
            max_range_km=max_range_km,
            range_step_km=range_step_km,
            min_elevation_deg=min_elevation_deg,
            beamwidth_deg=beamwidth_deg,
            freezing_level_m=freezing_level_m,
            clim_weight=clim_weight,
        )

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
                cappi_height + antenna_height_m,  # CAPPI height is above sea level
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
                    cappi_height + antenna_height_m,
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
                    )
                    elev_clim_corrections[i, j] = corr

            data_vars["elev_clim_correction_db"] = (
                ["range_km", "elevation"],
                elev_clim_corrections,
            )

            # Compute blended elevation corrections
            if quality_weight > 0:
                elev_blended = (
                    clim_weight * elev_clim_corrections + quality_weight * elev_corrections
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
            "cappi_correction_db": (["range_km", "cappi_height"], np.zeros((n_ranges, n_heights))),
            "cappi_beam_height_m": (["range_km", "cappi_height"], np.zeros((n_ranges, n_heights))),
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
                cappi_height + antenna_height_m,
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
                )
                elev_corrections[i, j] = corr
                elev_beam_heights[i, j] = beam_h

        data_vars["elev_correction_db"] = (["range_km", "elevation"], elev_corrections)
        data_vars["elev_beam_height_m"] = (["range_km", "elevation"], elev_beam_heights)
        data_vars["elev_clim_correction_db"] = (["range_km", "elevation"], elev_corrections)
        data_vars["elev_blended_correction_db"] = (["range_km", "elevation"], elev_corrections)
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
