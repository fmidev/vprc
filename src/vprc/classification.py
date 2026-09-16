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

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING
import logging

import numpy as np
import xarray as xr

from .constants import MDS, MIN_SAMPLES, EVAPORATION_THRESHOLD_DB, SNOW_FREEZING_LEVEL_THRESHOLD_M

if TYPE_CHECKING:
    from .bright_band import BrightBandResult

logger = logging.getLogger(__name__)


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


class PrecipitationType(Enum):
    """Precipitation phase/type classification.

    Perl: $sade values 2=snow, 3=sleet, 4=rain (lines 1334-1356).
    Based on bright band detection and freezing level height.
    """

    UNKNOWN = "unknown"  # Not precipitation or unclassified
    SNOW = "snow"        # No BB, freezing level < 500m
    SLEET = "sleet"      # BB detected, bottom at/below ground
    RAIN = "rain"        # BB with bottom above ground, or FL >= 500m


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
        precipitation_type: Phase classification (snow/sleet/rain) if precipitation
    """

    layers: list[EchoLayer] = field(default_factory=list)
    usable_for_vpr: bool = False
    profile_type: LayerType = LayerType.UNKNOWN
    freezing_level_m: float | None = None
    precipitation_type: PrecipitationType = PrecipitationType.UNKNOWN

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
    count: xr.DataArray | None,
    heights: np.ndarray,
    min_samples: int = MIN_SAMPLES,
) -> list[EchoLayer]:
    """Segment a 1D profile into contiguous echo layers.

    Layers are separated by missing corrected reflectivity. Sample counts
    have already been applied by smoothing; reapplying them would discard
    successfully interpolated gaps. ``count`` is retained for compatibility.

    Args:
        dbz: Reflectivity values along height dimension
        count: Sample counts along height dimension
        heights: Height coordinate values (sorted ascending)
        min_samples: Minimum sample count for valid data

    Returns:
        List of EchoLayer objects, ordered from lowest to highest
    """
    # Determine valid echo at each height
    valid = np.isfinite(dbz.values) & (dbz.values > MDS)

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
    third_to_sixth_difference: float | None = None,
) -> LayerType:
    """Classify a single echo layer.

    Apply the legacy layer-height, freezing-level and low-level-gradient gates.
    Clutter is retained as an independent flag; corrected precipitation may
    still contain clutter. Evaporation is checked only after BB processing.

    Args:
        layer: The EchoLayer to classify
        layer_index: 0-based index (0 = lowest layer)
        freezing_level_m: Freezing level in meters (None if unknown)
        lowest_profile_height: The lowest height in the profile
        has_clutter_flag: True if clutter was previously detected

    Returns:
        Classified LayerType
    """
    layer.has_clutter = has_clutter_flag and layer.bottom_height < 1000
    result = LayerType.GROUND_CLUTTER if layer.has_clutter else LayerType.UNKNOWN
    if layer_index > 0 or layer.bottom_height > 500:
        result = LayerType.ALTOSTRATUS
    elif layer.bottom_height < 500 and layer.top_height > 500:
        result = LayerType.CLEAR_AIR_ECHO
    if layer.touches_ground and layer.thickness <= 200:
        result = LayerType.CLEAR_AIR_ECHO

    # Unknown NWP retains the documented heuristic fallback, not an invented FL.
    if freezing_level_m is None:
        if layer_index == 0 and layer.touches_ground and layer.thickness >= 1000:
            return (
                LayerType.PRECIPITATION if layer.max_dbz >= 5
                else LayerType.CLEAR_AIR_ECHO
            )
        return result

    freezing_level = max(0.0, freezing_level_m)
    if (
        (not has_clutter_flag and layer_index == 0)
        or (has_clutter_flag and layer_index == 1)
    ):
        if (
            layer.bottom_height < 1500
            and layer.top_height < 3000
            and layer.max_dbz < 5
            and layer.top_height - freezing_level < 1000
        ):
            result = LayerType.CLEAR_AIR_ECHO

    if layer_index == 0:
        above_freezing = layer.top_height - freezing_level
        if (
            layer.touches_ground
            and above_freezing >= 1200
            and (
                freezing_level > 0
                or (
                    third_to_sixth_difference is not None
                    and third_to_sixth_difference <= 5.5
                    and layer.top_height > 1200
                )
            )
        ):
            result = LayerType.PRECIPITATION
        if layer.top_height < 2000 and (
            (above_freezing < 1200 and layer.touches_ground)
            or (
                third_to_sixth_difference is not None
                and third_to_sixth_difference >= 5.5
            )
        ):
            result = LayerType.CLEAR_AIR_ECHO
    return result


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
        ds: Previously smoothed xarray Dataset with 'corrected_dbz',
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
    count = ds.get("zcount")
    freezing_level = ds.attrs.get("freezing_level_m", None)
    lowest_height = int(heights[0]) if len(heights) > 0 else 0

    # Check if clutter was detected upstream (could be stored in attrs)
    has_clutter = ds.attrs.get("clutter_detected", False)
    third_to_sixth = None
    if lowest_height + 400 in heights and lowest_height + 1000 in heights:
        third_to_sixth = float(
            dbz.sel(height=lowest_height + 400)
            - dbz.sel(height=lowest_height + 1000)
        )

    # Find contiguous layers
    layers = _find_echo_layers(dbz, count, heights, min_samples)

    # Classify each layer
    for i, layer in enumerate(layers):
        layer.layer_type = _classify_layer(
            layer,
            layer_index=i,
            freezing_level_m=freezing_level,
            lowest_profile_height=lowest_height,
            has_clutter_flag=has_clutter,
            third_to_sixth_difference=third_to_sixth,
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
        ):
            usable_for_vpr = True

    return ProfileClassification(
        layers=layers,
        usable_for_vpr=usable_for_vpr,
        profile_type=profile_type,
        freezing_level_m=freezing_level,
    )


def finalize_profile_classification(
    classification: ProfileClassification,
    ds: xr.Dataset,
    bb_result: "BrightBandResult",
) -> ProfileClassification:
    """Refresh first-layer statistics and reject evaporation after BB correction.

    Call after restoring/capping the BB and applying post-BB clutter correction.
    Layer boundaries and preliminary layer types are retained. The BB enhancement
    is excluded from the evaporation maximum, as in ``$evayla = $zala``.
    Neither the input classification nor the dataset is modified.
    """
    result = replace(
        classification,
        layers=[replace(layer) for layer in classification.layers],
    )
    if not result.layers:
        return result
    layer = result.layers[0]
    upper = layer.top_height
    if (
        bb_result.detected
        and bb_result.bottom_height is not None
        and bb_result.bottom_height > 0
    ):
        upper = bb_result.bottom_height
    profile = ds["corrected_dbz"].sel(height=slice(layer.bottom_height, upper))
    valid = profile.where(np.isfinite(profile) & (profile > MDS), drop=True)
    layer.evaporation_detected = False
    layer.min_dbz_below_max = None
    layer.min_dbz_height = None
    if valid.size:
        max_index = int(np.argmax(valid.values))
        layer.max_dbz = float(valid.values[max_index])
        layer.max_dbz_height = int(valid.height.values[max_index])
        below = valid.sel(height=slice(None, layer.max_dbz_height))
        min_index = int(np.argmin(below.values))
        layer.min_dbz_below_max = float(below.values[min_index])
        layer.min_dbz_height = int(below.height.values[min_index])
    else:
        layer.max_dbz = MDS
        layer.max_dbz_height = None
        result.usable_for_vpr = False
    layer.has_clutter = bool(ds.attrs.get("clutter_detected", layer.has_clutter))
    if layer.layer_type == LayerType.PRECIPITATION and _check_evaporation(layer):
        layer.evaporation_detected = True
        layer.layer_type = LayerType.ALTOSTRATUS
        result.profile_type = LayerType.ALTOSTRATUS
        result.usable_for_vpr = False
        result.precipitation_type = PrecipitationType.UNKNOWN
    return result


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


def classify_precipitation_type(
    bb_result: "BrightBandResult",
    freezing_level_m: float | None,
    lowest_profile_height: int,
) -> PrecipitationType:
    """Classify precipitation phase based on bright band and freezing level.

    Determines whether precipitation is snow, sleet, or rain based on:
    - Bright band detection and position relative to antenna height zero
    - Freezing level height when no bright band is detected

    Perl logic (lines 1334-1356 allprof_prodx2.pl):
    - Snow: no BB and freezing level < 500m
    - Sleet: BB detected with bottom <= 0 (including surface sentinel -1)
    - Rain: BB with bottom above ground, or freezing level >= 500m without BB

    Args:
        bb_result: Bright band detection result
        freezing_level_m: Freezing level in meters above antenna (from NWP)
        lowest_profile_height: Retained for compatibility; positive observation
            heights must not be treated as the surface sentinel

    Returns:
        PrecipitationType indicating precipitation phase
    """
    # If freezing level is not available, cannot classify
    if freezing_level_m is None:
        logger.debug("Cannot classify precipitation type: no freezing level")
        return PrecipitationType.UNKNOWN

    if bb_result.detected:
        # BB detected - check if bottom is at/below ground
        bb_bottom = bb_result.bottom_height
        if bb_bottom is not None and bb_bottom <= 0:
            # BB extends to ground level - sleet/mixed precipitation
            logger.debug(
                "Precipitation type: SLEET (BB bottom %d <= zero; lowest %d)",
                bb_bottom,
                lowest_profile_height,
            )
            return PrecipitationType.SLEET
        else:
            # BB above ground - rain
            logger.debug(
                "Precipitation type: RAIN (BB bottom %s > zero; lowest %d)",
                bb_bottom,
                lowest_profile_height,
            )
            return PrecipitationType.RAIN
    else:
        # No BB detected - use freezing level threshold
        if freezing_level_m < SNOW_FREEZING_LEVEL_THRESHOLD_M:
            logger.debug(
                "Precipitation type: SNOW (no BB, FL %.0f < %d)",
                freezing_level_m,
                SNOW_FREEZING_LEVEL_THRESHOLD_M,
            )
            return PrecipitationType.SNOW
        else:
            logger.debug(
                "Precipitation type: RAIN (no BB, FL %.0f >= %d)",
                freezing_level_m,
                SNOW_FREEZING_LEVEL_THRESHOLD_M,
            )
            return PrecipitationType.RAIN
