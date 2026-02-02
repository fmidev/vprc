"""Spatial compositing of VPR corrections from multiple radars.

This module creates composite VPR correction grids by combining corrections
from multiple radar stations. The default weighting uses inverse distance
scaled by profile quality weight.

The weighting system is modular: different weight functions can be substituted
via the `weight_func` parameter.
"""

from dataclasses import dataclass
from typing import Protocol, Callable

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

from .vpr_correction import VPRCorrectionResult


# Type alias for weight functions
# Signature: (distances, quality_weights, **kwargs) -> weights
WeightFunc = Callable[..., np.ndarray]


class RadarLocation(Protocol):
    """Protocol for objects providing radar location."""

    latitude: float
    longitude: float


@dataclass
class RadarCorrection:
    """VPR correction result with associated radar metadata.

    Attributes:
        radar_code: Three-letter radar identifier (e.g., 'KAN')
        latitude: Radar latitude in WGS84 degrees
        longitude: Radar longitude in WGS84 degrees
        correction: VPR correction result from compute_vpr_correction()
        quality_weight: Profile quality weight (0-10 typical range)
    """

    radar_code: str
    latitude: float
    longitude: float
    correction: VPRCorrectionResult
    quality_weight: float


@dataclass
class CompositeGrid:
    """Regular grid definition for composite output.

    Attributes:
        x: 1D array of x coordinates (m) in target CRS
        y: 1D array of y coordinates (m) in target CRS
        crs: Coordinate reference system (pyproj CRS object)
    """

    x: np.ndarray
    y: np.ndarray
    crs: CRS

    @classmethod
    def from_bounds(
        cls,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        resolution_m: float = 1000.0,
        crs: CRS | int = 3067,
    ) -> "CompositeGrid":
        """Create grid from bounding box.

        Args:
            xmin, xmax, ymin, ymax: Bounding box in target CRS coordinates
            resolution_m: Grid cell size in meters
            crs: Target CRS (default EPSG:3067 ETRS-TM35FIN)

        Returns:
            CompositeGrid with regular spacing
        """
        if isinstance(crs, int):
            crs = CRS.from_epsg(crs)

        x = np.arange(xmin, xmax + resolution_m, resolution_m)
        y = np.arange(ymin, ymax + resolution_m, resolution_m)

        return cls(x=x, y=y, crs=crs)

    @classmethod
    def for_finland(
        cls, resolution_m: float = 1000.0, crs: CRS | int = 3067
    ) -> "CompositeGrid":
        """Create grid covering Finland's radar coverage area.

        Uses approximate bounds that cover the FMI radar network.

        Args:
            resolution_m: Grid cell size in meters
            crs: Target CRS (default EPSG:3067)

        Returns:
            CompositeGrid covering Finland
        """
        # Approximate ETRS-TM35FIN bounds for Finland radar coverage
        # Extends slightly beyond borders to cover max radar range
        return cls.from_bounds(
            xmin=50_000,
            xmax=750_000,
            ymin=6_600_000,
            ymax=7_800_000,
            resolution_m=resolution_m,
            crs=crs,
        )

    @classmethod
    def for_radars(
        cls,
        radar_codes: list[str] | None = None,
        range_km: float = 251.0,
        resolution_m: float = 1000.0,
        crs: CRS | int = 3067,
    ) -> "CompositeGrid":
        """Create grid covering specified radars with given range.

        Computes the bounding box that encompasses all specified radar
        locations plus a margin equal to range_km in all directions.

        Args:
            radar_codes: List of radar codes to include. If None, uses all
                configured radars with valid coordinates.
            range_km: Range around each radar to include (km). Default 251km
                covers 250km radar range plus 1km margin.
            resolution_m: Grid cell size in meters
            crs: Target CRS (default EPSG:3067 ETRS-TM35FIN)

        Returns:
            CompositeGrid covering all specified radars

        Raises:
            ValueError: If radar_codes is empty or no radars with valid
                coordinates are found
        """
        from .config import get_radar_coords, list_radar_codes

        if isinstance(crs, int):
            crs = CRS.from_epsg(crs)

        # Get radar codes if not specified
        if radar_codes is None:
            radar_codes = list_radar_codes()

        if not radar_codes:
            raise ValueError("radar_codes cannot be empty")

        # Collect coordinates (skip radars without coords)
        coords = []
        for code in radar_codes:
            coord = get_radar_coords(code)
            if coord is not None:
                coords.append(coord)

        if not coords:
            raise ValueError(
                f"No radars with valid coordinates found in: {radar_codes}"
            )

        # Transform from WGS84 to target CRS
        transformer = Transformer.from_crs(
            CRS.from_epsg(4326), crs, always_xy=True
        )

        xs, ys = [], []
        for lat, lon in coords:
            x, y = transformer.transform(lon, lat)  # lon, lat for always_xy
            xs.append(x)
            ys.append(y)

        # Compute bounds with range margin
        margin_m = range_km * 1000
        xmin = min(xs) - margin_m
        xmax = max(xs) + margin_m
        ymin = min(ys) - margin_m
        ymax = max(ys) + margin_m

        return cls.from_bounds(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            resolution_m=resolution_m,
            crs=crs,
        )


def inverse_distance_weight(
    distances_m: np.ndarray,
    quality_weights: np.ndarray,
    power: float = 2.0,
    min_distance_m: float = 1000.0,
    max_distance_m: float = 250_000.0,
) -> np.ndarray:
    """Compute inverse distance weights scaled by quality.

    Weight = quality_weight / max(distance, min_distance)^power

    Args:
        distances_m: Array of distances from radar to grid points (m)
        quality_weights: Array of quality weights (one per radar)
        power: IDW exponent (default 2.0)
        min_distance_m: Minimum distance to avoid division issues (m)

    Returns:
        Array of weights (same shape as distances_m)
    """
    # Clamp distances to minimum
    d = np.maximum(distances_m, min_distance_m)

    # IDW with quality scaling
    weights = quality_weights * (1 / np.power(d, power) - 1 / np.power(max_distance_m, power))

    return weights


def compute_radar_distances(
    grid: CompositeGrid,
    radars: list[RadarCorrection],
) -> np.ndarray:
    """Compute distances from each grid point to each radar.

    Args:
        grid: Target composite grid
        radars: List of radar corrections with locations

    Returns:
        Array of shape (n_radars, n_y, n_x) with distances in meters
    """
    # Transform radar locations from WGS84 to grid CRS
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), grid.crs, always_xy=True
    )

    # Create 2D grid of coordinates
    xx, yy = np.meshgrid(grid.x, grid.y)

    # Compute distances for each radar
    distances = np.zeros((len(radars), len(grid.y), len(grid.x)))

    for i, radar in enumerate(radars):
        rx, ry = transformer.transform(radar.longitude, radar.latitude)
        distances[i] = np.sqrt((xx - rx) ** 2 + (yy - ry) ** 2)

    return distances


def interpolate_correction_to_grid(
    correction: VPRCorrectionResult,
    distances_m: np.ndarray,
    correction_var: str = "cappi_blended_correction_db",
    cappi_height_m: int = 500,
) -> np.ndarray:
    """Interpolate range-dependent correction to 2D grid.

    For each grid point, uses the distance to radar to look up the
    appropriate correction value from the 1D range-dependent profile.

    Args:
        correction: VPR correction result
        distances_m: 2D array of distances from radar (m)
        correction_var: Name of correction variable to use
        cappi_height_m: CAPPI height to use for corrections (m)

    Returns:
        2D array of correction values (dB)
    """
    corr_ds = correction.corrections

    # Fall back to instant correction if blended not available
    if correction_var not in corr_ds:
        correction_var = "cappi_correction_db"

    # Get correction values and range coordinates
    # Dataset has dims: (range_km, cappi_height)
    range_km = corr_ds["range_km"].values
    cappi_heights = corr_ds["cappi_height"].values

    # Select the requested CAPPI height (or nearest)
    if cappi_height_m in cappi_heights:
        height_idx = np.where(cappi_heights == cappi_height_m)[0][0]
    else:
        height_idx = np.argmin(np.abs(cappi_heights - cappi_height_m))

    # Get 1D correction profile for this CAPPI height
    # Shape of corr_data: (n_range_bins, n_cappi_heights)
    corr_data = corr_ds[correction_var].values
    corr_1d = corr_data[:, height_idx]  # shape: (n_range_bins,)

    # Convert distances to km and interpolate
    dist_km = distances_m / 1000.0

    # Linear interpolation, extrapolate with edge values
    interp_corr = np.interp(
        dist_km.ravel(),
        range_km,
        corr_1d,
    ).reshape(distances_m.shape)

    return interp_corr


def create_empty_composite(grid: CompositeGrid, radar_codes: list[str] | None = None) -> xr.Dataset:
    """Create an empty composite Dataset with all NaN values.

    Used when no radars have usable VPR corrections (e.g., no precipitation).
    This produces a valid COG-compatible Dataset that can be written to GeoTIFF.

    Args:
        grid: Target composite grid
        radar_codes: Optional list of radar codes that were attempted

    Returns:
        xarray Dataset with:
        - correction_db: All NaN values
        - weight_sum: All zeros
        - n_radars: All zeros
    """
    ny, nx = len(grid.y), len(grid.x)

    return xr.Dataset(
        {
            "correction_db": (["y", "x"], np.full((ny, nx), np.nan)),
            "weight_sum": (["y", "x"], np.zeros((ny, nx))),
            "n_radars": (["y", "x"], np.zeros((ny, nx), dtype=np.int8)),
        },
        coords={
            "x": grid.x,
            "y": grid.y,
        },
        attrs={
            "crs": str(grid.crs),
            "crs_epsg": grid.crs.to_epsg(),
            "correction_variable": "none",
            "max_range_km": 0.0,
            "radar_codes": radar_codes or [],
            "empty_composite": True,
        },
    )


def composite_corrections(
    radars: list[RadarCorrection],
    grid: CompositeGrid,
    weight_func: WeightFunc = inverse_distance_weight,
    correction_var: str = "cappi_blended_correction_db",
    max_range_km: float = 250.0,
) -> xr.Dataset:
    """Create composite VPR correction grid from multiple radars.

    For areas covered by multiple radars, corrections are blended using
    the specified weight function. For areas covered by only one radar,
    that radar's correction is used directly.

    If no radars are provided (e.g., no precipitation detected by any radar),
    returns a valid empty composite with all NaN corrections. This is a valid
    operational scenario, not an error.

    Args:
        radars: List of RadarCorrection objects with VPR results (may be empty)
        grid: Target composite grid
        weight_func: Function to compute weights from distances and quality
        correction_var: Name of correction variable to composite
        max_range_km: Maximum range to use corrections (km)

    Returns:
        xarray Dataset with:
        - correction_db: Composite VPR correction (dB), NaN where no coverage
        - weight_sum: Total weight at each point (for QC)
        - n_radars: Number of radars contributing at each point
    """
    # Handle empty radar list gracefully (no precipitation scenario)
    if not radars:
        return create_empty_composite(grid)

    n_radars = len(radars)
    ny, nx = len(grid.y), len(grid.x)

    # Compute distances from each grid point to each radar
    distances = compute_radar_distances(grid, radars)

    # Get quality weights as array
    quality_weights = np.array([r.quality_weight for r in radars])

    # Compute weights for each radar at each grid point
    weights = np.zeros((n_radars, ny, nx))
    corrections = np.zeros((n_radars, ny, nx))

    for i, radar in enumerate(radars):
        # Interpolate this radar's correction to the grid
        corrections[i] = interpolate_correction_to_grid(
            radar.correction, distances[i], correction_var
        )

        # Compute weights (quality scaled by distance)
        w = weight_func(distances[i], quality_weights[i], max_distance_m=max_range_km * 1000)

        # Zero weight beyond max range
        w[distances[i] > max_range_km * 1000] = 0.0

        # Zero weight if quality weight is zero (unusable profile)
        if quality_weights[i] <= 0:
            w[:] = 0.0

        weights[i] = w

    # Weighted average
    weight_sum = np.sum(weights, axis=0)
    n_contributing = np.sum(weights > 0, axis=0)

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        composite = np.sum(weights * corrections, axis=0) / weight_sum

    # Set no-coverage areas to NaN
    composite[weight_sum == 0] = np.nan

    # Build output Dataset
    ds = xr.Dataset(
        {
            "correction_db": (["y", "x"], composite),
            "weight_sum": (["y", "x"], weight_sum),
            "n_radars": (["y", "x"], n_contributing.astype(np.int8)),
        },
        coords={
            "x": grid.x,
            "y": grid.y,
        },
        attrs={
            "crs": str(grid.crs),
            "crs_epsg": grid.crs.to_epsg(),
            "correction_variable": correction_var,
            "max_range_km": max_range_km,
            "radar_codes": [r.radar_code for r in radars],
            "empty_composite": False,
        },
    )

    return ds


def create_radar_correction(
    radar_code: str,
    latitude: float,
    longitude: float,
    correction: VPRCorrectionResult,
) -> RadarCorrection:
    """Create RadarCorrection from correction result.

    Convenience function that extracts quality_weight from the correction.

    Args:
        radar_code: Three-letter radar identifier
        latitude: Radar latitude (WGS84 degrees)
        longitude: Radar longitude (WGS84 degrees)
        correction: VPR correction result

    Returns:
        RadarCorrection with all fields populated
    """
    return RadarCorrection(
        radar_code=radar_code,
        latitude=latitude,
        longitude=longitude,
        correction=correction,
        quality_weight=correction.quality_weight,
    )
