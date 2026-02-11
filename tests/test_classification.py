"""Tests for the classification module."""

import numpy as np
import pytest
import xarray as xr

from vprc.classification import (
    EchoLayer,
    LayerType,
    PrecipitationType,
    ProfileClassification,
    _find_echo_layers,
    _classify_layer,
    _check_evaporation,
    classify_profile,
    classify_profiles,
    classify_precipitation_type,
)
from vprc.constants import MDS, MIN_SAMPLES, EVAPORATION_THRESHOLD_DB, SNOW_FREEZING_LEVEL_THRESHOLD_M
from vprc.bright_band import BrightBandResult


def make_test_dataset(
    dbz_values: list[float],
    heights: list[int] | None = None,
    counts: list[int] | None = None,
    freezing_level_m: float | None = None,
) -> xr.Dataset:
    """Create a minimal test dataset for classification tests."""
    if heights is None:
        heights = list(range(100, 100 + 200 * len(dbz_values), 200))
    if counts is None:
        counts = [100] * len(dbz_values)

    ds = xr.Dataset(
        {
            "corrected_dbz": ("height", dbz_values),
            "sample_count": ("height", counts),
        },
        coords={"height": heights},
        attrs={"freezing_level_m": freezing_level_m},
    )
    return ds


class TestEchoLayer:
    """Tests for EchoLayer dataclass."""

    def test_thickness_calculation(self):
        layer = EchoLayer(bottom_height=500, top_height=2000)
        assert layer.thickness == 1500

    def test_default_values(self):
        layer = EchoLayer(bottom_height=100, top_height=300)
        assert layer.max_dbz == MDS
        assert layer.max_dbz_height is None
        assert layer.layer_type == LayerType.UNKNOWN
        assert layer.touches_ground is False
        assert layer.has_clutter is False
        assert layer.evaporation_detected is False

    def test_evaporation_amount(self):
        """Evaporation amount is computed from max and min below max."""
        layer = EchoLayer(
            bottom_height=100,
            top_height=1000,
            max_dbz=40.0,
            max_dbz_height=800,
            min_dbz_below_max=15.0,
            min_dbz_height=300,
        )
        assert layer.evaporation_amount == 25.0

    def test_evaporation_amount_none_without_min(self):
        """Evaporation amount is None if min not available."""
        layer = EchoLayer(
            bottom_height=100,
            top_height=1000,
            max_dbz=40.0,
            max_dbz_height=800,
        )
        assert layer.evaporation_amount is None


class TestProfileClassification:
    """Tests for ProfileClassification dataclass."""

    def test_empty_profile(self):
        result = ProfileClassification()
        assert result.num_layers == 0
        assert result.lowest_layer is None
        assert result.usable_for_vpr is False

    def test_with_layers(self):
        layers = [
            EchoLayer(bottom_height=100, top_height=2000, layer_type=LayerType.PRECIPITATION),
            EchoLayer(bottom_height=3000, top_height=4000, layer_type=LayerType.ALTOSTRATUS),
        ]
        result = ProfileClassification(layers=layers, usable_for_vpr=True)
        assert result.num_layers == 2
        assert result.lowest_layer == layers[0]


class TestFindEchoLayers:
    """Tests for layer segmentation."""

    def test_single_continuous_layer(self):
        """Profile with echo at all levels forms one layer."""
        dbz = xr.DataArray([10, 15, 20, 15, 10], dims=["height"])
        count = xr.DataArray([100, 100, 100, 100, 100], dims=["height"])
        heights = np.array([100, 300, 500, 700, 900])

        layers = _find_echo_layers(dbz, count, heights)

        assert len(layers) == 1
        assert layers[0].bottom_height == 100
        assert layers[0].top_height == 900
        assert layers[0].max_dbz == 20
        assert layers[0].max_dbz_height == 500
        assert layers[0].touches_ground == True

    def test_two_layers_with_gap(self):
        """Profile with gap in middle forms two layers."""
        dbz = xr.DataArray([10, 15, MDS, MDS, 20, 25], dims=["height"])
        count = xr.DataArray([100, 100, 100, 100, 100, 100], dims=["height"])
        heights = np.array([100, 300, 500, 700, 900, 1100])

        layers = _find_echo_layers(dbz, count, heights)

        assert len(layers) == 2
        assert layers[0].bottom_height == 100
        assert layers[0].top_height == 300
        assert layers[0].touches_ground == True
        assert layers[1].bottom_height == 900
        assert layers[1].top_height == 1100
        assert layers[1].touches_ground == False

    def test_no_echo(self):
        """Profile with all MDS values produces no layers."""
        dbz = xr.DataArray([MDS, MDS, MDS], dims=["height"])
        count = xr.DataArray([100, 100, 100], dims=["height"])
        heights = np.array([100, 300, 500])

        layers = _find_echo_layers(dbz, count, heights)

        assert len(layers) == 0

    def test_low_sample_count_invalidates_layer(self):
        """Points with insufficient samples are treated as gaps."""
        dbz = xr.DataArray([10, 15, 20], dims=["height"])
        count = xr.DataArray([100, 10, 100], dims=["height"])  # Middle has low count
        heights = np.array([100, 300, 500])

        layers = _find_echo_layers(dbz, count, heights)

        assert len(layers) == 2  # Two separate layers
        assert layers[0].top_height == 100
        assert layers[1].bottom_height == 500

    def test_elevated_single_layer(self):
        """Layer starting above ground is not marked as touching_ground."""
        dbz = xr.DataArray([MDS, MDS, 20, 25, 20], dims=["height"])
        count = xr.DataArray([100, 100, 100, 100, 100], dims=["height"])
        heights = np.array([100, 300, 500, 700, 900])

        layers = _find_echo_layers(dbz, count, heights)

        assert len(layers) == 1
        assert layers[0].bottom_height == 500
        assert layers[0].touches_ground == False


class TestClassifyLayer:
    """Tests for individual layer classification."""

    def test_thin_ground_layer_is_clear_air(self):
        """Very thin ground layer (200m) classified as clear air."""
        layer = EchoLayer(
            bottom_height=100, top_height=300, max_dbz=10, touches_ground=True
        )
        result = _classify_layer(layer, 0, freezing_level_m=2000, lowest_profile_height=100)
        assert result == LayerType.CLEAR_AIR_ECHO

    def test_elevated_layer_is_altostratus(self):
        """Layer detached from ground is altostratus."""
        layer = EchoLayer(
            bottom_height=3000, top_height=5000, max_dbz=20, touches_ground=False
        )
        result = _classify_layer(layer, 1, freezing_level_m=2000, lowest_profile_height=100)
        assert result == LayerType.ALTOSTRATUS

    def test_thick_ground_layer_above_fl_is_precipitation(self):
        """Thick ground layer extending well above freezing level is precipitation."""
        layer = EchoLayer(
            bottom_height=100, top_height=4000, max_dbz=30, touches_ground=True
        )
        result = _classify_layer(layer, 0, freezing_level_m=2000, lowest_profile_height=100)
        # top - FL = 4000 - 2000 = 2000 >= 1200
        assert result == LayerType.PRECIPITATION

    def test_weak_thick_layer_is_clear_air(self):
        """Thick but weak (max_dbz < 5) ground layer is clear air."""
        layer = EchoLayer(
            bottom_height=100, top_height=2000, max_dbz=3, touches_ground=True
        )
        result = _classify_layer(layer, 0, freezing_level_m=None, lowest_profile_height=100)
        assert result == LayerType.CLEAR_AIR_ECHO

    def test_clutter_flag_overrides(self):
        """Clutter flag from upstream processing takes precedence."""
        layer = EchoLayer(
            bottom_height=100, top_height=3000, max_dbz=30, touches_ground=True
        )
        result = _classify_layer(
            layer, 0, freezing_level_m=2000, lowest_profile_height=100, has_clutter_flag=True
        )
        assert result == LayerType.GROUND_CLUTTER


class TestClassifyProfile:
    """Tests for full profile classification."""

    def test_precipitation_profile(self):
        """Profile with thick ground-attached layer above FL is precipitation."""
        # Create profile extending from ground to well above freezing level
        dbz_values = [20, 25, 30, 28, 25, 22, 20, 18, 15, 12, 10]
        heights = list(range(100, 100 + 200 * len(dbz_values), 200))
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=1000)

        result = classify_profile(ds)

        assert result.num_layers == 1
        assert result.profile_type == LayerType.PRECIPITATION
        assert result.usable_for_vpr == True
        assert result.lowest_layer.touches_ground == True

    def test_clear_air_echo_profile(self):
        """Low, thin layer is classified as clear air."""
        dbz_values = [5, 3, MDS, MDS, MDS]
        ds = make_test_dataset(dbz_values, freezing_level_m=2000)

        result = classify_profile(ds)

        assert result.num_layers == 1
        assert result.profile_type == LayerType.CLEAR_AIR_ECHO
        assert result.usable_for_vpr == False

    def test_empty_profile(self):
        """Profile with no echo."""
        dbz_values = [MDS, MDS, MDS, MDS]
        ds = make_test_dataset(dbz_values)

        result = classify_profile(ds)

        assert result.num_layers == 0
        assert result.profile_type == LayerType.UNKNOWN
        assert result.usable_for_vpr == False

    def test_altostratus_profile(self):
        """Elevated layer without ground echo is altostratus."""
        dbz_values = [MDS, MDS, MDS, 20, 25, 20, 15]
        heights = [100, 300, 500, 2000, 2200, 2400, 2600]
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=1500)

        result = classify_profile(ds)

        assert result.num_layers == 1
        assert result.profile_type == LayerType.ALTOSTRATUS
        assert result.usable_for_vpr == False

    def test_multiple_layers(self):
        """Profile with ground layer and elevated layer."""
        dbz_values = [15, 20, 25, 20, MDS, MDS, 18, 22, 18]
        heights = [100, 300, 500, 700, 900, 1100, 3000, 3200, 3400]
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=None)

        result = classify_profile(ds)

        assert result.num_layers == 2
        # First layer is thick enough for precipitation
        assert result.lowest_layer.thickness == 600


class TestClassifyProfiles:
    """Tests for multi-time classification."""

    def test_single_time_fallback(self):
        """Dataset without time dimension returns single-element list."""
        ds = make_test_dataset([20, 25, 30, 25, 20], freezing_level_m=1000)

        results = classify_profiles(ds)

        assert len(results) == 1
        assert isinstance(results[0], ProfileClassification)

    def test_multiple_times(self):
        """Dataset with time dimension returns classification per time."""
        heights = [100, 300, 500, 700, 900]
        dbz_t0 = [20, 25, 30, 25, 20]
        dbz_t1 = [MDS, MDS, 15, 20, 15]  # Elevated layer

        ds = xr.Dataset(
            {
                "corrected_dbz": (["time", "height"], [dbz_t0, dbz_t1]),
                "sample_count": (["time", "height"], [[100] * 5, [100] * 5]),
            },
            coords={"height": heights, "time": [0, 1]},
            attrs={"freezing_level_m": 400},
        )

        results = classify_profiles(ds)

        assert len(results) == 2
        # First time: ground layer
        assert results[0].lowest_layer.touches_ground == True
        # Second time: elevated layer
        assert results[1].lowest_layer.touches_ground == False


class TestEvaporationCheck:
    """Tests for evaporation detection.

    Perl logic (lines 1174-1185): If dBZ drops by >20 from max to min below max,
    precipitation is evaporating before reaching the surface → reclassify as altostratus.
    """

    def test_no_evaporation_when_max_at_bottom(self):
        """No evaporation if max dBZ is at or below min height."""
        layer = EchoLayer(
            bottom_height=100,
            top_height=1000,
            max_dbz=40.0,
            max_dbz_height=100,  # Max at bottom
            min_dbz_below_max=None,
            min_dbz_height=None,
        )
        assert _check_evaporation(layer) is False

    def test_no_evaporation_with_small_drop(self):
        """No evaporation if dBZ drop is within threshold."""
        layer = EchoLayer(
            bottom_height=100,
            top_height=1000,
            max_dbz=30.0,
            max_dbz_height=800,
            min_dbz_below_max=15.0,  # 15 dB drop, below 20 threshold
            min_dbz_height=300,
        )
        assert _check_evaporation(layer) is False
        assert layer.evaporation_amount == 15.0

    def test_evaporation_detected_with_large_drop(self):
        """Evaporation detected if dBZ drop exceeds 20 dB."""
        layer = EchoLayer(
            bottom_height=100,
            top_height=1000,
            max_dbz=40.0,
            max_dbz_height=800,
            min_dbz_below_max=15.0,  # 25 dB drop, exceeds 20 threshold
            min_dbz_height=300,
        )
        assert _check_evaporation(layer) is True
        assert layer.evaporation_amount == 25.0

    def test_evaporation_exactly_at_threshold_not_detected(self):
        """Evaporation not detected if exactly at 20 dB threshold."""
        layer = EchoLayer(
            bottom_height=100,
            top_height=1000,
            max_dbz=40.0,
            max_dbz_height=800,
            min_dbz_below_max=20.0,  # Exactly 20 dB drop
            min_dbz_height=300,
        )
        assert _check_evaporation(layer) is False

    def test_evaporation_reclassifies_to_altostratus(self):
        """Profile with strong evaporation is classified as altostratus."""
        # Create profile with max at top and low values at bottom
        # Strong drop: 45 dBZ at 900m, 10 dBZ at 100m = 35 dB drop
        dbz_values = [10, 15, 20, 30, 45]
        heights = [100, 300, 500, 700, 900]
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=1500)

        result = classify_profile(ds)

        # Should be classified as altostratus due to evaporation
        assert result.num_layers == 1
        assert result.lowest_layer.evaporation_detected is True
        assert result.profile_type == LayerType.ALTOSTRATUS
        assert result.usable_for_vpr is False

    def test_profile_without_evaporation_is_precipitation(self):
        """Profile with normal gradient is classified as precipitation."""
        # Create profile with max near bottom (no evaporation)
        # Values decrease upward as expected
        dbz_values = [30, 28, 25, 22, 18, 15, 12, 10]
        heights = [100, 300, 500, 700, 900, 1100, 1300, 1500]
        ds = make_test_dataset(dbz_values, heights, freezing_level_m=800)

        result = classify_profile(ds)

        assert result.num_layers == 1
        assert result.lowest_layer.evaporation_detected is False
        # With freezing level at 800m and top at 1500m (700m above FL),
        # this should be precipitation
        assert result.lowest_layer.touches_ground

    def test_find_echo_layers_computes_min_below_max(self):
        """_find_echo_layers correctly identifies min dBZ below max."""
        # Profile: 10, 15, 40, 25, 20 (max at 500m, min below at 100m)
        dbz_values = [10.0, 15.0, 40.0, 25.0, 20.0]
        heights = [100, 300, 500, 700, 900]
        counts = [100] * 5
        ds = make_test_dataset(dbz_values, heights, counts)

        dbz = ds["corrected_dbz"]
        count = ds["sample_count"]
        layers = _find_echo_layers(dbz, count, np.array(heights))

        assert len(layers) == 1
        layer = layers[0]
        assert layer.max_dbz == 40.0
        assert layer.max_dbz_height == 500
        assert layer.min_dbz_below_max == 10.0
        assert layer.min_dbz_height == 100


class TestClassifyPrecipitationType:
    """Tests for precipitation phase classification.

    Perl logic (lines 1334-1356 allprof_prodx2.pl):
    - Snow ($sade=2): no BB and freezing level < 500m
    - Sleet ($sade=3): BB detected with bottom at/below ground
    - Rain ($sade=4): BB above ground or freezing level >= 500m without BB
    """

    def test_snow_no_bb_low_freezing_level(self):
        """Snow when no BB detected and freezing level < 500m."""
        bb = BrightBandResult(detected=False)
        result = classify_precipitation_type(
            bb, freezing_level_m=300, lowest_profile_height=100
        )
        assert result == PrecipitationType.SNOW

    def test_snow_boundary_freezing_level(self):
        """Snow at threshold boundary (< 500m)."""
        bb = BrightBandResult(detected=False)
        result = classify_precipitation_type(
            bb, freezing_level_m=499, lowest_profile_height=100
        )
        assert result == PrecipitationType.SNOW

    def test_rain_no_bb_high_freezing_level(self):
        """Rain when no BB and freezing level >= 500m."""
        bb = BrightBandResult(detected=False)
        result = classify_precipitation_type(
            bb, freezing_level_m=500, lowest_profile_height=100
        )
        assert result == PrecipitationType.RAIN

    def test_rain_no_bb_well_above_threshold(self):
        """Rain when no BB and freezing level well above threshold."""
        bb = BrightBandResult(detected=False)
        result = classify_precipitation_type(
            bb, freezing_level_m=2000, lowest_profile_height=100
        )
        assert result == PrecipitationType.RAIN

    def test_sleet_bb_at_ground(self):
        """Sleet when BB bottom is at ground level."""
        bb = BrightBandResult(
            detected=True,
            peak_height=300,
            bottom_height=100,  # Same as lowest_profile_height
            top_height=500,
        )
        result = classify_precipitation_type(
            bb, freezing_level_m=600, lowest_profile_height=100
        )
        assert result == PrecipitationType.SLEET

    def test_sleet_bb_below_ground(self):
        """Sleet when BB bottom would be below profile lowest level."""
        bb = BrightBandResult(
            detected=True,
            peak_height=200,
            bottom_height=50,  # Below lowest profile height
            top_height=400,
        )
        result = classify_precipitation_type(
            bb, freezing_level_m=500, lowest_profile_height=100
        )
        assert result == PrecipitationType.SLEET

    def test_rain_bb_above_ground(self):
        """Rain when BB is clearly above ground."""
        bb = BrightBandResult(
            detected=True,
            peak_height=1500,
            bottom_height=1100,  # Well above ground
            top_height=1900,
        )
        result = classify_precipitation_type(
            bb, freezing_level_m=2000, lowest_profile_height=100
        )
        assert result == PrecipitationType.RAIN

    def test_rain_bb_just_above_ground(self):
        """Rain when BB bottom is just above ground level."""
        bb = BrightBandResult(
            detected=True,
            peak_height=500,
            bottom_height=300,  # One step above ground
            top_height=700,
        )
        result = classify_precipitation_type(
            bb, freezing_level_m=800, lowest_profile_height=100
        )
        assert result == PrecipitationType.RAIN

    def test_unknown_when_no_freezing_level(self):
        """Unknown type when freezing level is not available."""
        bb = BrightBandResult(detected=True, peak_height=1000)
        result = classify_precipitation_type(
            bb, freezing_level_m=None, lowest_profile_height=100
        )
        assert result == PrecipitationType.UNKNOWN

    def test_bb_detected_but_no_bottom_height(self):
        """Rain when BB detected but bottom_height is None."""
        bb = BrightBandResult(
            detected=True,
            peak_height=1000,
            bottom_height=None,  # Not determined
            top_height=1200,
        )
        result = classify_precipitation_type(
            bb, freezing_level_m=1500, lowest_profile_height=100
        )
        # Without bottom height, cannot determine sleet, defaults to rain
        assert result == PrecipitationType.RAIN

    def test_threshold_matches_constant(self):
        """Verify threshold matches the constant value."""
        assert SNOW_FREEZING_LEVEL_THRESHOLD_M == 500
