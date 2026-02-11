"""Echo layer classification and profile quality assessment.

This module segments vertical profiles into contiguous echo layers and
classifies them for downstream VPR correction.

Based on allprof_prodx2.pl section "Kerrosten tunnistus ja tunnuslukujen
poiminta" (layer detection and feature extraction).

Layer types:
    - PRECIPITATION: Continuous echo reaching ground, suitable for VPR correction
    - ALTOSTRATUS: Elevated layer detached from ground
    - CLEAR_AIR_ECHO: Boundary layer echo (insects, turbulence) - not precipitation
    - GROUND_CLUTTER: Ground clutter contaminated echo
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
import xarray as xr

from .constants import MDS, MIN_SAMPLES, EVAPORATION_THRESHOLD_DB


class LayerType(Enum):
    """Classification of echo layers.

    Perl equivalents: $sade (precipitation), $askerros (altostratus),
    $rajakkaiku (clear-air echo), $maakaiku (ground clutter).
    """

    PRECIPITATION = "precipitation"
    ALTOSTRATUS = "altostratus"
    CLEAR_AIR_ECHO = "clear_air_echo"
    GROUND_CLUTTER = "ground_clutter"
    UNKNOWN = "unknown"


@dataclass
class EchoLayer:
    """A contiguous vertical layer of radar echo.

    Attributes:
        bottom_height: Lower bound of the layer (m above antenna)
        top_height: Upper bound of the layer (m above antenna)
        max_dbz: Maximum reflectivity within the layer (dBZ)
        max_dbz_height: Height of maximum reflectivity (m)
        min_dbz_below_max: Minimum reflectivity between ground and max height (dBZ)
        min_dbz_height: Height of minimum reflectivity below max (m)
        layer_type: Classification of the layer
        touches_ground: True if layer starts at the lowest profile level
        has_clutter: True if ground clutter was detected in this layer
        evaporation_detected: True if significant evaporation detected (>20dB drop)
    """

    bottom_height: int
    top_height: int
    max_dbz: float = MDS
    max_dbz_height: int | None = None
    min_dbz_below_max: float | None = None
    min_dbz_height: int | None = None
    layer_type: LayerType = LayerType.UNKNOWN
    touches_ground: bool = False
    has_clutter: bool = False
    evaporation_detected: bool = False

    @property
    def thickness(self) -> int:
        """Layer thickness in meters."""
        return self.top_height - self.bottom_height

    @property
    def evaporation_amount(self) -> float | None:
        """dBZ drop from max to min below max, if calculable."""
        if self.min_dbz_below_max is not None and self.max_dbz > MDS:
            return self.max_dbz - self.min_dbz_below_max
        return None


@dataclass
class ProfileClassification:
    """Classification results for a complete vertical profile.

    Attributes:
        layers: List of detected echo layers, ordered by height
        usable_for_vpr: True if profile is suitable for VPR correction
        profile_type: Summary classification based on lowest layer
        freezing_level_m: Freezing level used for classification (if available)
    """

    layers: list[EchoLayer] = field(default_factory=list)
    usable_for_vpr: bool = False
    profile_type: LayerType = LayerType.UNKNOWN
    freezing_level_m: float | None = None

    @property
    def num_layers(self) -> int:
        """Number of detected echo layers."""
        return len(self.layers)

    @property
    def lowest_layer(self) -> EchoLayer | None:
        """The lowest echo layer, if any."""
        return self.layers[0] if self.layers else None


def _find_echo_layers(
    dbz: xr.DataArray,
    count: xr.DataArray,
    heights: np.ndarray,
    min_samples: int = MIN_SAMPLES,
) -> list[EchoLayer]:
    """Segment a 1D profile into contiguous echo layers.

    Layers are separated by gaps (heights with dbz <= MDS or insufficient samples).

    Args:
        dbz: Reflectivity values along height dimension
        count: Sample counts along height dimension
        heights: Height coordinate values (sorted ascending)
        min_samples: Minimum sample count for valid data

    Returns:
        List of EchoLayer objects, ordered from lowest to highest
    """
    # Determine valid echo at each height
    valid = (dbz.values > MDS) & (count.values >= min_samples)

    layers = []
    in_layer = False
    current_bottom = None
    current_max_dbz = MDS
    current_max_height = None
    current_max_idx = None
    layer_start_idx = None
    lowest_height = heights[0] if len(heights) > 0 else None

    for i, h in enumerate(heights):
        if valid[i]:
            if not in_layer:
                # Start new layer
                in_layer = True
                current_bottom = int(h)
                layer_start_idx = i
                current_max_dbz = float(dbz.values[i])
                current_max_height = int(h)
                current_max_idx = i
            else:
                # Continue layer, update max
                if dbz.values[i] > current_max_dbz:
                    current_max_dbz = float(dbz.values[i])
                    current_max_height = int(h)
                    current_max_idx = i
        else:
            if in_layer:
                # End current layer at previous height
                top_height = int(heights[i - 1])
                layer = _create_layer_with_min_dbz(
                    dbz=dbz,
                    heights=heights,
                    valid=valid,
                    bottom_height=current_bottom,
                    top_height=top_height,
                    max_dbz=current_max_dbz,
                    max_dbz_height=current_max_height,
                    max_idx=current_max_idx,
                    layer_start_idx=layer_start_idx,
                    lowest_height=lowest_height,
                )
                layers.append(layer)
                in_layer = False

    # Close final layer if profile ends in echo
    if in_layer:
        top_height = int(heights[-1])
        layer = _create_layer_with_min_dbz(
            dbz=dbz,
            heights=heights,
            valid=valid,
            bottom_height=current_bottom,
            top_height=top_height,
            max_dbz=current_max_dbz,
            max_dbz_height=current_max_height,
            max_idx=current_max_idx,
            layer_start_idx=layer_start_idx,
            lowest_height=lowest_height,
        )
        layers.append(layer)

    return layers


def _create_layer_with_min_dbz(
    dbz: xr.DataArray,
    heights: np.ndarray,
    valid: np.ndarray,
    bottom_height: int,
    top_height: int,
    max_dbz: float,
    max_dbz_height: int,
    max_idx: int,
    layer_start_idx: int,
    lowest_height: int | None,
) -> EchoLayer:
    """Create an EchoLayer, computing min dBZ below max height.

    The min dBZ below max is used for evaporation detection.

    Args:
        dbz: Full dBZ array
        heights: Full heights array
        valid: Boolean array of valid heights
        bottom_height: Layer bottom
        top_height: Layer top
        max_dbz: Maximum dBZ in layer
        max_dbz_height: Height of maximum
        max_idx: Index of maximum dBZ
        layer_start_idx: Starting index of layer
        lowest_height: Lowest height in profile (for touches_ground)

    Returns:
        EchoLayer with min_dbz_below_max computed
    """
    min_dbz_below_max = None
    min_dbz_height = None

    # Find minimum dBZ between layer bottom and max height
    # Only meaningful if max is above layer bottom
    if max_idx > layer_start_idx:
        min_val = float("inf")
        for idx in range(layer_start_idx, max_idx):
            if valid[idx] and dbz.values[idx] < min_val:
                min_val = float(dbz.values[idx])
                min_dbz_height = int(heights[idx])
        if min_val < float("inf"):
            min_dbz_below_max = min_val

    return EchoLayer(
        bottom_height=bottom_height,
        top_height=top_height,
        max_dbz=max_dbz,
        max_dbz_height=max_dbz_height,
        min_dbz_below_max=min_dbz_below_max,
        min_dbz_height=min_dbz_height,
        touches_ground=(bottom_height == lowest_height),
    )


def _classify_layer(
    layer: EchoLayer,
    layer_index: int,
    freezing_level_m: float | None,
    lowest_profile_height: int,
    has_clutter_flag: bool = False,
) -> LayerType:
    """Classify a single echo layer.

    Classification logic derived from allprof_prodx2.pl:
    - Ground-attached thick layers with echo above freezing level → PRECIPITATION
    - Thin ground layers or weak echo near boundary layer → CLEAR_AIR
    - Elevated layers detached from ground → ALTOSTRATUS
    - Layers with detected ground clutter → CLUTTER
    - Layers with significant evaporation (>20dB drop) → ALTOSTRATUS

    Args:
        layer: The EchoLayer to classify
        layer_index: 0-based index (0 = lowest layer)
        freezing_level_m: Freezing level in meters (None if unknown)
        lowest_profile_height: The lowest height in the profile
        has_clutter_flag: True if clutter was previously detected

    Returns:
        Classified LayerType
    """
    # Clutter flag from upstream processing takes precedence
    if has_clutter_flag and layer.touches_ground:
        layer.has_clutter = True
        return LayerType.GROUND_CLUTTER

    # Elevated layers (not touching ground or not the lowest) are altostratus
    if layer_index > 0 or layer.bottom_height > lowest_profile_height + 400:
        return LayerType.ALTOSTRATUS

    # Ground-attached layer analysis
    thickness = layer.thickness

    # Very thin ground layer (≤200m) is likely clutter or CAE
    if thickness <= 200:
        return LayerType.CLEAR_AIR_ECHO

    # Check for significant evaporation (Perl lines 1174-1185)
    # If dBZ drops by >20 from max to min below max, precipitation is evaporating
    if _check_evaporation(layer):
        layer.evaporation_detected = True
        return LayerType.ALTOSTRATUS

    # Check if layer extends well above freezing level (precipitation signature)
    if freezing_level_m is not None and freezing_level_m > 0:
        height_above_fl = layer.top_height - freezing_level_m
        if height_above_fl >= 1200:
            return LayerType.PRECIPITATION
        if height_above_fl < 0 and layer.top_height < 2000:
            return LayerType.CLEAR_AIR_ECHO

    # Thick ground-attached layer with significant extent
    if layer.touches_ground and thickness >= 1000:
        # Weak echo suggests CAE even if thick
        if layer.max_dbz < 5:
            return LayerType.CLEAR_AIR_ECHO
        return LayerType.PRECIPITATION

    # Moderate thickness without clear precipitation signature
    if layer.top_height < 800:
        return LayerType.CLEAR_AIR_ECHO

    # Default for ambiguous cases: treat as unknown/needs more info
    return LayerType.UNKNOWN


def _check_evaporation(layer: EchoLayer) -> bool:
    """Check if layer shows significant evaporation.

    Evaporation is detected when reflectivity drops significantly from
    the profile maximum toward the ground, indicating precipitation
    is not reaching the surface.

    Perl logic (lines 1174-1185 allprof_prodx2.pl):
        if (max_height > min_height and profile_is_precipitation):
            evapor = dBZ_at_max - dBZ_at_min_below_max
            if evapor > 20:
                not_precipitation (reclassify as altostratus)

    Args:
        layer: EchoLayer with max_dbz and min_dbz_below_max computed

    Returns:
        True if significant evaporation detected (>20dB drop)
    """
    # Evaporation check requires:
    # 1. Max dBZ height above min dBZ height (max not at bottom)
    # 2. Both values are valid
    if layer.max_dbz_height is None or layer.min_dbz_height is None:
        return False

    if layer.max_dbz_height <= layer.min_dbz_height:
        return False

    evap_amount = layer.evaporation_amount
    if evap_amount is None:
        return False

    return evap_amount > EVAPORATION_THRESHOLD_DB


def classify_profile(
    ds: xr.Dataset,
    min_samples: int = MIN_SAMPLES,
) -> ProfileClassification:
    """Classify a vertical profile into echo layers.

    Segments the corrected_dbz field into contiguous layers and classifies
    each layer based on its characteristics and position.

    Args:
        ds: xarray Dataset with 'corrected_dbz' and 'sample_count' variables,
            indexed by 'height'. Must have 'freezing_level_m' in attrs
            (can be None).

    Returns:
        ProfileClassification with detected layers and quality assessment

    Example:
        >>> ds = read_vvp("profile.txt")
        >>> ds = apply_clutter_correction(ds)
        >>> ds = apply_spike_smoothing(ds)
        >>> classification = classify_profile(ds)
        >>> classification.usable_for_vpr
        True
        >>> classification.profile_type
        <LayerType.PRECIPITATION: 'precipitation'>
    """
    heights = ds["height"].values
    dbz = ds["corrected_dbz"]
    count = ds["sample_count"]
    freezing_level = ds.attrs.get("freezing_level_m", None)
    lowest_height = int(heights[0]) if len(heights) > 0 else 0

    # Check if clutter was detected upstream (could be stored in attrs)
    has_clutter = ds.attrs.get("clutter_detected", False)

    # Find contiguous layers
    layers = _find_echo_layers(dbz, count, heights, min_samples)

    # Classify each layer
    for i, layer in enumerate(layers):
        layer.layer_type = _classify_layer(
            layer,
            layer_index=i,
            freezing_level_m=freezing_level,
            lowest_profile_height=lowest_height,
            has_clutter_flag=has_clutter and i == 0,
        )

    # Determine overall profile classification
    profile_type = LayerType.UNKNOWN
    usable_for_vpr = False

    if layers:
        lowest = layers[0]
        profile_type = lowest.layer_type

        # VPR correction is useful for precipitation profiles
        # that touch the ground and have sufficient vertical extent
        if (
            lowest.layer_type == LayerType.PRECIPITATION
            and lowest.touches_ground
            and lowest.thickness >= 800
        ):
            usable_for_vpr = True

    return ProfileClassification(
        layers=layers,
        usable_for_vpr=usable_for_vpr,
        profile_type=profile_type,
        freezing_level_m=freezing_level,
    )


def classify_profiles(
    ds: xr.Dataset,
    time_dim: str = "time",
    min_samples: int = MIN_SAMPLES,
) -> list[ProfileClassification]:
    """Classify multiple profiles along a time dimension.

    Convenience function for datasets with multiple time steps.

    Args:
        ds: Dataset with 'height' and 'time' dimensions
        time_dim: Name of the time dimension
        min_samples: Minimum sample count for valid data

    Returns:
        List of ProfileClassification, one per time step
    """
    if time_dim not in ds.dims:
        return [classify_profile(ds, min_samples)]

    results = []
    for t in ds[time_dim].values:
        ds_t = ds.sel({time_dim: t})
        results.append(classify_profile(ds_t, min_samples))

    return results
