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
    BEAMWIDTH_DEG,
    MAX_CORRECTION_DB,
    MIN_CORRECTION_THRESHOLD_DB,
    DEFAULT_CAPPI_HEIGHTS_M,
    DEFAULT_MAX_RANGE_KM,
    DEFAULT_RANGE_STEP_KM,
)


@dataclass
class VPRCorrectionResult:
    """Results of VPR correction calculation.

    Attributes:
        corrections: xarray Dataset with correction factors vs range
        z_ground_dbz: Reference reflectivity at ground level (dBZ)
        usable: True if corrections were successfully computed
    """

    corrections: xr.Dataset
    z_ground_dbz: float
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
    beamwidth_deg: float = BEAMWIDTH_DEG,
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
    beamwidth_deg: float = BEAMWIDTH_DEG,
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
    beamwidth_deg: float = BEAMWIDTH_DEG,
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


def compute_vpr_correction(
    ds: xr.Dataset,
    cappi_heights_m: tuple[int, ...] | None = None,
    max_range_km: int = DEFAULT_MAX_RANGE_KM,
    range_step_km: int = DEFAULT_RANGE_STEP_KM,
    min_elevation_deg: float = 0.5,
    beamwidth_deg: float = BEAMWIDTH_DEG,
) -> VPRCorrectionResult:
    """Compute VPR correction factors vs range for CAPPI heights.

    This is the main entry point for VPR correction calculation.

    Args:
        ds: Dataset with 'corrected_dbz' indexed by 'height'.
            Must have 'antenna_height_m' in attrs.
        cappi_heights_m: CAPPI heights to compute corrections for.
            Default: (500, 1000)
        max_range_km: Maximum range for corrections (default 250 km)
        range_step_km: Range step size (default 1 km)
        min_elevation_deg: Minimum elevation angle for pseudo-CAPPI
        beamwidth_deg: Radar beamwidth (degrees)

    Returns:
        VPRCorrectionResult containing:
        - corrections: Dataset with dims (range_km, cappi_height)
        - z_ground_dbz: Reference ground reflectivity
        - usable: Whether corrections are valid

    Example:
        >>> ds = read_vvp("profile.txt")
        >>> ds = remove_ground_clutter(ds)
        >>> ds = smooth_spikes(ds)
        >>> result = compute_vpr_correction(ds)
        >>> result.corrections['correction_db'].sel(cappi_height=500, range_km=100)
    """
    if cappi_heights_m is None:
        cappi_heights_m = DEFAULT_CAPPI_HEIGHTS_M

    antenna_height_m = ds.attrs.get("antenna_height_m", 0)

    # Interpolate profile to fine grid
    fine_heights, fine_z = interpolate_to_fine_grid(ds)

    # Determine ground reference Z (lowest valid level)
    heights = ds["height"].values
    dbz_values = ds["corrected_dbz"].values
    valid_mask = dbz_values > MDS

    if not np.any(valid_mask):
        # No valid echo - return empty result
        return _create_empty_result(cappi_heights_m, max_range_km, range_step_km)

    lowest_valid_idx = np.argmax(valid_mask)
    z_ground_dbz = float(dbz_values[lowest_valid_idx])
    z_ground = idecibel(z_ground_dbz)

    # Create range array
    ranges_km = np.arange(range_step_km, max_range_km + range_step_km, range_step_km)
    n_ranges = len(ranges_km)
    n_heights = len(cappi_heights_m)

    # Allocate output arrays
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

    # Build output Dataset
    correction_ds = xr.Dataset(
        data_vars={
            "correction_db": (["range_km", "cappi_height"], corrections),
            "beam_height_m": (["range_km", "cappi_height"], beam_heights),
        },
        coords={
            "range_km": ranges_km,
            "cappi_height": list(cappi_heights_m),
        },
        attrs={
            "radar": ds.attrs.get("radar", "unknown"),
            "timestamp": str(ds.attrs.get("timestamp", "")),
            "z_ground_dbz": z_ground_dbz,
            "antenna_height_m": antenna_height_m,
            "beamwidth_deg": beamwidth_deg,
            "min_elevation_deg": min_elevation_deg,
        },
    )

    return VPRCorrectionResult(
        corrections=correction_ds,
        z_ground_dbz=z_ground_dbz,
        usable=True,
    )


def _create_empty_result(
    cappi_heights_m: tuple[int, ...],
    max_range_km: int,
    range_step_km: int,
) -> VPRCorrectionResult:
    """Create an empty VPR correction result when no valid echo exists."""
    ranges_km = np.arange(range_step_km, max_range_km + range_step_km, range_step_km)
    n_ranges = len(ranges_km)
    n_heights = len(cappi_heights_m)

    correction_ds = xr.Dataset(
        data_vars={
            "correction_db": (["range_km", "cappi_height"], np.zeros((n_ranges, n_heights))),
            "beam_height_m": (["range_km", "cappi_height"], np.zeros((n_ranges, n_heights))),
        },
        coords={
            "range_km": ranges_km,
            "cappi_height": list(cappi_heights_m),
        },
    )

    return VPRCorrectionResult(
        corrections=correction_ds,
        z_ground_dbz=MDS,
        usable=False,
    )
