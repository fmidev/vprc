"""Tests for climatological VPR profile generation."""

import numpy as np
import pytest

from vprc.climatology import generate_climatological_profile, get_clim_ground_reference
from vprc.constants import MDS, STEP


class TestClimatologicalProfile:
    """Tests for generate_climatological_profile()."""

    def test_base_value_calculation(self):
        """Base dBZ should be 10 + FL_km * 10."""
        # FL = 2000m -> base = 10 + 2.0 * 10 = 30 dBZ
        ds = generate_climatological_profile(freezing_level_m=2000)
        assert ds.attrs["base_dbz"] == pytest.approx(30.0)

        # FL = 3500m -> base = 10 + 3.5 * 10 = 45 dBZ
        ds = generate_climatological_profile(freezing_level_m=3500)
        assert ds.attrs["base_dbz"] == pytest.approx(45.0)

        # FL = 500m -> base = 10 + 0.5 * 10 = 15 dBZ
        ds = generate_climatological_profile(freezing_level_m=500)
        assert ds.attrs["base_dbz"] == pytest.approx(15.0)

    def test_melting_layer_enhancement(self):
        """Melting layer should have +3.5/+7.0/+3.5 dB structure."""
        ds = generate_climatological_profile(
            freezing_level_m=2000,
            lowest_level_m=100,
        )
        base = 30.0  # 10 + 2.0 * 10

        # Below BB (FL-600 = 1400m): base value
        assert ds["clim_dbz"].sel(height=1500, method="nearest").values == pytest.approx(base)

        # Lower BB (FL-400 to FL-200): +3.5 dB
        # At 1700m (between 1600 and 1800)
        val_1700 = ds["clim_dbz"].sel(height=1700, method="nearest").values
        assert val_1700 == pytest.approx(base + 3.5)

        # BB peak (FL-200 to FL): +7.0 dB
        # At 1900m (between 1800 and 2000)
        val_1900 = ds["clim_dbz"].sel(height=1900, method="nearest").values
        assert val_1900 == pytest.approx(base + 7.0)

        # Upper BB (FL to FL+200): +3.5 dB
        # At 2100m (between 2000 and 2200)
        val_2100 = ds["clim_dbz"].sel(height=2100, method="nearest").values
        assert val_2100 == pytest.approx(base + 3.5)

    def test_decay_above_melting_layer(self):
        """Above FL+400, profile should decay at -0.94 dB per 200m."""
        ds = generate_climatological_profile(
            freezing_level_m=2000,
            lowest_level_m=100,
        )
        base = 30.0

        # At FL+300 = 2300m: back to base (between FL+200 and FL+400)
        val_2300 = ds["clim_dbz"].sel(height=2300, method="nearest").values
        assert val_2300 == pytest.approx(base)

        # At FL+500 = 2500m: base - 0.94 (first decay step)
        val_2500 = ds["clim_dbz"].sel(height=2500, method="nearest").values
        assert val_2500 == pytest.approx(base - 0.94)

        # At FL+700 = 2700m: base - 2*0.94
        val_2700 = ds["clim_dbz"].sel(height=2700, method="nearest").values
        assert val_2700 == pytest.approx(base - 2 * 0.94)

    def test_negative_freezing_level(self):
        """When FL < 0, decay from surface at -0.66 dB per 200m."""
        ds = generate_climatological_profile(
            freezing_level_m=-500,
            lowest_level_m=100,
        )
        # base = 10 + (-0.5) * 10 = 5 dBZ
        base = 5.0

        # Lowest level: base
        val_100 = ds["clim_dbz"].sel(height=100, method="nearest").values
        assert val_100 == pytest.approx(base)

        # 200m higher: base - 0.66
        val_300 = ds["clim_dbz"].sel(height=300, method="nearest").values
        assert val_300 == pytest.approx(base - 0.66)

        # 400m higher: base - 2*0.66
        val_500 = ds["clim_dbz"].sel(height=500, method="nearest").values
        assert val_500 == pytest.approx(base - 2 * 0.66)

    def test_mds_floor(self):
        """Profile values should never go below MDS (-45 dBZ)."""
        # Very negative FL -> low base, should hit MDS
        ds = generate_climatological_profile(
            freezing_level_m=-3000,
            lowest_level_m=100,
            max_height_m=5000,
        )
        # base = 10 + (-3.0) * 10 = -20 dBZ
        # After decay, should hit MDS

        # All values should be >= MDS
        assert (ds["clim_dbz"].values >= MDS).all()

    def test_profile_shape(self):
        """Profile should have correct dimensions and coordinates."""
        ds = generate_climatological_profile(
            freezing_level_m=2000,
            lowest_level_m=100,
            max_height_m=5000,
            step_m=200,
        )

        assert "height" in ds.coords
        assert "clim_dbz" in ds.data_vars
        assert ds["height"].values[0] == 100
        assert ds["height"].values[-1] == 4900  # Last value before 5000
        # Heights: 100, 300, 500, ... 4900 = 25 levels
        assert len(ds["height"]) == 25

    def test_freezing_level_stored_in_attrs(self):
        """Freezing level should be stored in dataset attributes."""
        ds = generate_climatological_profile(freezing_level_m=1850)
        assert ds.attrs["freezing_level_m"] == 1850


class TestClimGroundReference:
    """Tests for get_clim_ground_reference()."""

    def test_elevated_bb(self):
        """When BB is elevated, ground reference is base value."""
        ref = get_clim_ground_reference(freezing_level_m=2000, lowest_level_m=100)
        # base = 10 + 2.0 * 10 = 30 dBZ
        assert ref == pytest.approx(30.0)

    def test_bb_at_ground(self):
        """When BB is at ground, ground reference is still base (no BB enhancement)."""
        ref = get_clim_ground_reference(freezing_level_m=300, lowest_level_m=100)
        # base = 10 + 0.3 * 10 = 13 dBZ (not enhanced)
        assert ref == pytest.approx(13.0)

    def test_negative_fl(self):
        """When FL < 0, ground reference is base value."""
        ref = get_clim_ground_reference(freezing_level_m=-500, lowest_level_m=100)
        # base = 10 + (-0.5) * 10 = 5 dBZ
        assert ref == pytest.approx(5.0)
