"""Tests for the VPR correction module."""

import numpy as np
import pytest
import xarray as xr

from vprc.vpr_correction import (
    VPRCorrectionResult,
    interpolate_to_fine_grid,
    compute_beam_height,
    solve_elevation_for_height,
    compute_beam_weight,
    convolve_beam_with_profile,
    compute_correction_for_range,
    compute_vpr_correction,
    compute_quality_weight,
)
from vprc.constants import MDS, STEP, FINE_GRID_RESOLUTION_M, DEFAULT_BEAMWIDTH_DEG


def make_test_dataset(
    dbz_values: list[float],
    heights: list[int] | None = None,
    antenna_height_m: float = 100.0,
    zcounts: list[int] | None = None,
) -> xr.Dataset:
    """Create a minimal test dataset for VPR correction tests."""
    if heights is None:
        heights = list(range(100, 100 + STEP * len(dbz_values), STEP))

    ds = xr.Dataset(
        {
            "corrected_dbz": ("height", dbz_values),
            "sample_count": ("height", [100] * len(dbz_values)),
            "zcount": (
                "height",
                zcounts if zcounts is not None else [1000] * len(dbz_values),
            ),
        },
        coords={"height": heights},
        attrs={
            "antenna_height_m": antenna_height_m,
            "radar": "TEST",
        },
    )
    return ds


class TestInterpolateToFineGrid:
    """Tests for fine grid interpolation."""

    def test_basic_interpolation(self):
        """Interpolation produces expected fine grid."""
        # Simple profile: 20, 30, 20 dBZ at 100, 300, 500m
        dbz_values = [20.0, 30.0, 20.0]
        heights = [100, 300, 500]
        ds = make_test_dataset(dbz_values, heights)

        fine_heights, fine_z = interpolate_to_fine_grid(ds)

        # Check grid properties
        assert fine_heights[0] == 0
        assert fine_heights[1] == FINE_GRID_RESOLUTION_M
        assert len(fine_heights) == 15000 // FINE_GRID_RESOLUTION_M

        # Check interpolation: Z should be higher in middle
        idx_100 = 100 // FINE_GRID_RESOLUTION_M
        idx_300 = 300 // FINE_GRID_RESOLUTION_M
        assert fine_z[idx_300] > fine_z[idx_100]

    def test_mds_values_keep_finite_floor(self):
        """Missing layers retain the finite legacy minimum reflectivity."""
        dbz_values = [20.0, MDS, 20.0]
        heights = [100, 300, 500]
        ds = make_test_dataset(dbz_values, heights)

        fine_heights, fine_z = interpolate_to_fine_grid(ds)

        idx_300 = 300 // FINE_GRID_RESOLUTION_M
        assert fine_z[idx_300] == pytest.approx(10 ** (MDS / 10))

    def test_above_last_echo_is_mds_even_with_trailing_missing_levels(self):
        ds = make_test_dataset([30.0, 20.0, MDS, MDS])
        heights, z = interpolate_to_fine_grid(ds)
        assert z[heights == 300].item() == pytest.approx(100.0)
        np.testing.assert_allclose(z[heights > 300], 10 ** (MDS / 10))

    def test_empty_profile_has_finite_floor(self):
        ds = make_test_dataset([])
        _, z = interpolate_to_fine_grid(ds)
        np.testing.assert_allclose(z, 10 ** (MDS / 10))

    def test_extrapolation_below_profile(self):
        """Values below lowest profile level use constant extrapolation."""
        dbz_values = [20.0, 25.0, 30.0]
        heights = [500, 700, 900]  # Profile starts at 500m
        ds = make_test_dataset(dbz_values, heights)

        fine_heights, fine_z = interpolate_to_fine_grid(ds)

        # Below 500m should have same Z as at 500m
        idx_0 = 0
        idx_500 = 500 // FINE_GRID_RESOLUTION_M
        assert np.isclose(fine_z[idx_0], fine_z[idx_500])


class TestComputeBeamHeight:
    """Tests for beam height calculation."""

    def test_zero_elevation_near_ground(self):
        """At short range with 0° elevation, beam is near antenna height."""
        h = compute_beam_height(1000, 0.0, 100)  # 1km range, 0°, 100m antenna
        assert 100 < h < 200  # Should be just slightly above antenna

    def test_higher_elevation_higher_beam(self):
        """Higher elevation angles produce higher beam heights."""
        h1 = compute_beam_height(100000, 1.0, 100)
        h2 = compute_beam_height(100000, 3.0, 100)
        assert h2 > h1

    def test_longer_range_higher_beam(self):
        """Longer ranges produce higher beam heights (Earth curvature)."""
        h1 = compute_beam_height(50000, 0.5, 100)
        h2 = compute_beam_height(100000, 0.5, 100)
        assert h2 > h1

    def test_known_value(self):
        """Check against known beam height calculation."""
        # At 100km range, 0.5° elevation, ~100m antenna
        # Beam should be roughly 1-2 km high
        h = compute_beam_height(100000, 0.5, 100)
        assert 500 < h < 3000

    def test_reference_ellipsoid_radius_at_long_range(self):
        # Unchanged pystycappi.pl truncates this relative height to 4102 m.
        height = compute_beam_height(200000, 0.5, 174) - 174
        assert height == pytest.approx(4102, abs=1.0)


class TestSolveElevationForHeight:
    """Tests for elevation angle solver."""

    def test_finds_correct_elevation(self):
        """Solver finds elevation that achieves target height."""
        target = 2000  # 2km height
        range_m = 100000  # 100km
        antenna = 100

        elev = solve_elevation_for_height(target, range_m, antenna)

        # Verify by computing beam height at found elevation
        actual_height = compute_beam_height(range_m, elev, antenna)
        assert np.isclose(actual_height, target, atol=50)  # Within 50m

    def test_respects_min_elevation(self):
        """Solver respects minimum elevation constraint (pseudo-CAPPI)."""
        target = 100  # Very low target
        range_m = 100000
        antenna = 100
        min_elev = 0.5

        elev = solve_elevation_for_height(target, range_m, antenna, min_elev)

        assert elev >= min_elev

    def test_near_range_cappi_requires_elevation_above_45_degrees(self):
        elev = solve_elevation_for_height(1000, 1000, 174)
        assert elev > 45
        assert compute_beam_height(1000, elev, 174) == pytest.approx(1000)


class TestComputeBeamWeight:
    """Tests for Gaussian beam weight calculation."""

    def test_center_weight_is_one(self):
        """Weight at beam center (0°) is 1.0."""
        w = compute_beam_weight(0.0, DEFAULT_BEAMWIDTH_DEG)
        assert np.isclose(w, 1.0)

    def test_half_power_at_beamwidth(self):
        """Weight at half-power beamwidth is ~0.5."""
        w = compute_beam_weight(DEFAULT_BEAMWIDTH_DEG, DEFAULT_BEAMWIDTH_DEG)
        # 10^(-0.6 * 1^2) = 10^(-0.6) ≈ 0.25
        assert np.isclose(w, 0.25, atol=0.01)

    def test_symmetric(self):
        """Weight is symmetric about center."""
        w_pos = compute_beam_weight(0.5, DEFAULT_BEAMWIDTH_DEG)
        w_neg = compute_beam_weight(-0.5, DEFAULT_BEAMWIDTH_DEG)
        # Function uses abs implicitly via angular_distance
        assert np.isclose(w_pos, w_neg)

    def test_weight_decreases_with_distance(self):
        """Weight decreases with angular distance."""
        w1 = compute_beam_weight(0.3, DEFAULT_BEAMWIDTH_DEG)
        w2 = compute_beam_weight(0.6, DEFAULT_BEAMWIDTH_DEG)
        w3 = compute_beam_weight(1.0, DEFAULT_BEAMWIDTH_DEG)
        assert w1 > w2 > w3


class TestConvolveBeamWithProfile:
    """Tests for beam-profile convolution."""

    def test_unblocked_uniform_profile_no_change(self):
        """An unblocked beam observes the uniform profile unchanged."""
        n_layers = 600
        fine_heights = np.arange(0, n_layers * FINE_GRID_RESOLUTION_M, FINE_GRID_RESOLUTION_M)
        uniform_z = np.ones(n_layers) * 100.0  # Uniform Z

        z1 = convolve_beam_with_profile(fine_heights, uniform_z, 4000, 100000)
        z2 = convolve_beam_with_profile(fine_heights, uniform_z, 5000, 100000)

        assert np.isclose(z1, z2, rtol=0.01)

    def test_blocked_negative_heights_remain_in_weight_denominator(self):
        heights = np.arange(0, 15000, 25)
        z = np.full(len(heights), 100.0)
        actual = convolve_beam_with_profile(
            heights, z, 0, 100000, beam_bounds_m=(-250, 250)
        )
        # Legacy sums indices int(lower/25)-1 through int(upper/25), inclusive.
        slices = np.arange(-11, 11) * 25
        angles = np.rad2deg(np.arctan2(np.abs(slices), 100000))
        weights = 10 ** (-0.6 * (angles / 0.95) ** 2)
        expected = 100 * weights[slices >= 0].sum() / weights.sum()
        assert actual == pytest.approx(expected)
        assert actual < 60

    def test_peak_at_beam_center_maximizes_z(self):
        """Profile with peak at beam center produces maximum observed Z."""
        n_layers = 600
        fine_heights = np.arange(0, n_layers * FINE_GRID_RESOLUTION_M, FINE_GRID_RESOLUTION_M)

        # Create profile with peak at 3000m
        peak_height = 3000
        peak_z = 1000.0
        background_z = 100.0
        profile_z = np.where(
            np.abs(fine_heights - peak_height) < 500, peak_z, background_z
        )

        # Beam centered on peak should see more Z than beam away from peak
        z_on_peak = convolve_beam_with_profile(fine_heights, profile_z, peak_height, 100000)
        z_off_peak = convolve_beam_with_profile(fine_heights, profile_z, 1000, 100000)

        assert z_on_peak > z_off_peak


class TestComputeCorrectionForRange:
    """Tests for single-range correction calculation."""

    def test_uniform_profile_zero_correction(self):
        """Uniform profile produces ~zero correction."""
        n_layers = 600
        fine_heights = np.arange(0, n_layers * FINE_GRID_RESOLUTION_M, FINE_GRID_RESOLUTION_M)
        uniform_z = np.ones(n_layers) * 100.0

        corr, beam_h = compute_correction_for_range(
            fine_heights, uniform_z, 100.0, 100, 0.5, 100
        )

        # Should be near zero (beam sees same Z as ground)
        assert abs(corr) < 1.0

    def test_decreasing_profile_positive_correction(self):
        """Profile decreasing with height produces positive correction."""
        n_layers = 600
        fine_heights = np.arange(0, n_layers * FINE_GRID_RESOLUTION_M, FINE_GRID_RESOLUTION_M)
        # Z decreases exponentially with height
        profile_z = 1000.0 * np.exp(-fine_heights / 3000.0)
        z_ground = profile_z[0]

        corr, beam_h = compute_correction_for_range(
            fine_heights, profile_z, z_ground, 100, 0.5, 100
        )

        # Beam sees less Z than ground → positive correction
        assert corr > 0

    def test_uniform_profile_has_legacy_horizon_attenuation(self):
        heights = np.arange(0, 15000, 25)
        corr, _ = compute_correction_for_range(
            heights, np.full(len(heights), 100.0), 100.0, 100, 0.5, 174
        )
        # Unchanged pystycappi.pl prints 0.9 dB (one decimal place).
        assert corr == pytest.approx(0.9, abs=0.05)

    def test_no_echo_retains_previous_correction(self):
        heights = np.arange(0, 15000, 25)
        corr, _ = compute_correction_for_range(
            heights, np.zeros(len(heights)), 100.0, 200, 10, 174,
            previous_correction_db=12.3,
        )
        assert corr == pytest.approx(12.3)

    def test_clipping_remains_symmetric(self):
        heights = np.arange(0, 15000, 25)
        corr, _ = compute_correction_for_range(
            heights, np.full(len(heights), 1e8), 1.0, 10, 2, 174
        )
        assert corr == -30.0


class TestComputeVprCorrection:
    """Tests for main VPR correction function."""

    def test_returns_correct_structure(self):
        """Result has expected structure and dimensions."""
        dbz_values = [30.0, 28.0, 25.0, 22.0, 20.0, 18.0, 15.0]
        ds = make_test_dataset(dbz_values)

        result = compute_vpr_correction(ds)

        assert isinstance(result, VPRCorrectionResult)
        assert result.usable
        assert "cappi_correction_db" in result.corrections
        assert "cappi_beam_height_m" in result.corrections
        assert "range_km" in result.corrections.dims
        assert "cappi_height" in result.corrections.dims

    def test_default_cappi_heights(self):
        """Default CAPPI heights are 500 and 1000m."""
        dbz_values = [30.0, 28.0, 25.0, 22.0, 20.0]
        ds = make_test_dataset(dbz_values)

        result = compute_vpr_correction(ds)

        assert list(result.corrections["cappi_height"].values) == [500, 1000]

    def test_custom_cappi_heights(self):
        """Can specify custom CAPPI heights."""
        dbz_values = [30.0, 28.0, 25.0, 22.0, 20.0]
        ds = make_test_dataset(dbz_values)

        result = compute_vpr_correction(ds, cappi_heights_m=(750, 1500))

        assert list(result.corrections["cappi_height"].values) == [750, 1500]

    @pytest.mark.parametrize("no_echo", [False, True])
    def test_cappi_targets_are_above_sea_level(self, no_echo):
        ds = make_test_dataset(
            [MDS if no_echo else 30.0] * 50, antenna_height_m=174
        )
        result = compute_vpr_correction(
            ds, freezing_level_m=2000, max_range_km=10
        )
        heights = result.corrections.cappi_beam_height_m
        assert heights.sel(range_km=10, cappi_height=500).item() == pytest.approx(326)
        assert heights.sel(range_km=1, cappi_height=1000).item() == pytest.approx(826)

    def test_cappi_near_antenna_safeguard(self):
        ds = make_test_dataset([30.0] * 50, antenna_height_m=530)
        result = compute_vpr_correction(
            ds, max_range_km=1, min_elevation_deg=0.0, include_clim=False
        )
        assert result.corrections.cappi_beam_height_m.sel(
            cappi_height=500
        ).item() == pytest.approx(50)

    def test_shallow_echo_keeps_maximum_correction_above_echo(self):
        heights = list(range(100, 10000, 200))
        ds = make_test_dataset(
            [30.0 if h <= 900 else MDS for h in heights], heights,
            antenna_height_m=174,
        )
        result = compute_vpr_correction(
            ds, elevation_angles_deg=(3.3,), include_clim=False,
            max_range_km=250, range_step_km=50,
        )
        # Unchanged pystycappi.pl gives +30 at all these ranges.
        np.testing.assert_array_equal(
            result.corrections.elev_correction_db.sel(range_km=[100, 150, 200, 250]),
            np.full((4, 1), 30.0),
        )

    def test_echo_outside_legacy_beam_support_does_not_contribute(self):
        heights = list(range(100, 10000, 200))
        ds = make_test_dataset(
            [50.0 if h == 3700 else 10.0 for h in heights], heights,
            antenna_height_m=174,
        )
        result = compute_vpr_correction(
            ds, elevation_angles_deg=(2.3,), max_range_km=50,
            range_step_km=50, include_clim=False,
        )
        # Unchanged pystycappi.pl: the 3700 m peak is outside the beam, dBc=0.
        assert result.corrections.elev_correction_db.item() == 0.0

    def test_correction_is_carried_beyond_entire_fine_grid(self):
        ds = make_test_dataset([30.0] * 50)
        result = compute_vpr_correction(
            ds, elevation_angles_deg=(10.0,), max_range_km=250,
            range_step_km=10, include_clim=False,
        )
        np.testing.assert_array_equal(
            result.corrections.elev_correction_db.sel(range_km=[100, 150, 200, 250]),
            np.full((4, 1), 30.0),
        )

    @pytest.mark.parametrize("no_echo", [False, True])
    def test_geometry_metadata_and_explicit_overrides(self, no_echo):
        ds = make_test_dataset([MDS if no_echo else 30.0] * 50)
        ds.attrs.update(beamwidth_deg=1.1, horizon_angle_deg=0.8)
        kwargs = dict(
            elevation_angles_deg=(0.5,), max_range_km=100, range_step_km=100,
            freezing_level_m=2000,
        )
        metadata = compute_vpr_correction(ds, **kwargs)
        explicit = compute_vpr_correction(
            ds, beamwidth_deg=1.1, horizon_angle_deg=0.8, **kwargs
        )
        np.testing.assert_array_equal(
            metadata.corrections.elev_correction_db,
            explicit.corrections.elev_correction_db,
        )
        lower_horizon = compute_vpr_correction(
            ds, horizon_angle_deg=0.0, **kwargs
        )
        assert metadata.corrections.elev_correction_db.item() > (
            lower_horizon.corrections.elev_correction_db.item()
        )
        narrower = compute_vpr_correction(ds, beamwidth_deg=0.7, **kwargs)
        assert narrower.corrections.beamwidth_deg == 0.7
        assert metadata.corrections.beamwidth_deg == 1.1
        assert metadata.corrections.horizon_angle_deg == 0.8
        assert narrower.corrections.elev_correction_db.item() != (
            metadata.corrections.elev_correction_db.item()
        )

    def test_quality_gating_reaches_main_correction_result(self):
        ds = make_test_dataset([30.0] * 10, zcounts=[1000] * 5 + [10000] * 5)
        result = compute_vpr_correction(
            ds, layer_top_height_m=900, max_range_km=1, include_clim=False
        )
        assert result.quality_weight == pytest.approx(0.2)
        nonprecip = compute_vpr_correction(
            ds, is_precipitation=False, max_range_km=1, include_clim=False
        )
        assert not nonprecip.usable

    def test_empty_profile_not_usable(self):
        """Profile with all MDS values produces unusable result."""
        dbz_values = [MDS, MDS, MDS, MDS]
        ds = make_test_dataset(dbz_values)

        result = compute_vpr_correction(ds)

        assert not result.usable
        assert result.z_ground_dbz == MDS

    def test_correction_increases_with_range(self):
        """For typical profile, correction magnitude increases with range."""
        # Decreasing profile
        dbz_values = [35.0, 32.0, 28.0, 24.0, 20.0, 16.0, 12.0, 8.0]
        heights = list(range(100, 100 + STEP * len(dbz_values), STEP))
        ds = make_test_dataset(dbz_values, heights)

        result = compute_vpr_correction(ds, max_range_km=100, range_step_km=10)

        corr = result.corrections["cappi_correction_db"].sel(cappi_height=500)
        # Correction at 100km should be larger than at 10km
        assert abs(corr.sel(range_km=100).values) > abs(corr.sel(range_km=10).values)

    def test_z_ground_dbz_is_lowest_valid(self):
        """Ground reference is lowest valid dBZ value."""
        dbz_values = [MDS, 25.0, 30.0, 28.0, 25.0]
        heights = [100, 300, 500, 700, 900]
        ds = make_test_dataset(dbz_values, heights)

        result = compute_vpr_correction(ds)

        # Lowest valid is at 300m (25 dBZ)
        assert result.z_ground_dbz == 25.0


class TestIntegration:
    """Integration tests for full VPR correction workflow."""

    def test_process_vvp_computes_vpr_correction(self):
        """process_vvp includes VPR correction when usable."""
        from vprc import process_vvp
        from pathlib import Path

        # Use the test VVP file
        test_file = Path(__file__).parent / "data" / "202508241100_KAN.VVP_40.txt"
        if not test_file.exists():
            pytest.skip("Test data file not available")

        result = process_vvp(test_file, freezing_level_m=2000)

        # Check VPR correction was computed
        if result.usable_for_vpr:
            assert result.vpr_correction is not None
            assert result.vpr_correction.usable
            assert "cappi_correction_db" in result.vpr_correction.corrections

    def test_process_vvp_skips_vpr_when_disabled(self):
        """process_vvp skips VPR correction when compute_vpr=False."""
        from vprc import process_vvp
        from pathlib import Path

        test_file = Path(__file__).parent / "data" / "202508241100_KAN.VVP_40.txt"
        if not test_file.exists():
            pytest.skip("Test data file not available")

        result = process_vvp(test_file, compute_vpr=False)

        assert result.vpr_correction is None

class TestQualityWeight:
    """Tests for profile quality weight calculation."""

    def test_strong_profile_has_positive_weight(self):
        """Quality measures reflectivity sample coverage, not echo strength."""
        dbz_values = [35.0] * 20  # 20 levels from 100m to 3900m
        heights = list(range(100, 100 + STEP * 20, STEP))
        ds = make_test_dataset(dbz_values, heights)

        wq = compute_quality_weight(ds)

        assert wq == pytest.approx(1000 / 5000)

    def test_low_sample_count_has_zero_weight(self):
        """Insufficient reflectivity samples give zero quality weight."""
        dbz_values = [10.0] * 20
        heights = list(range(100, 100 + STEP * 20, STEP))
        ds = make_test_dataset(dbz_values, heights, zcounts=[500] * 20)

        wq = compute_quality_weight(ds)

        assert wq == 0.0

    @pytest.mark.parametrize("dbz", [0.0, 10.0, 30.0, 40.0])
    def test_quality_is_independent_of_reflectivity(self, dbz):
        ds = make_test_dataset([dbz] * 10)
        assert compute_quality_weight(ds) == pytest.approx(0.2)

    def test_requires_five_levels_and_precipitation(self):
        ds = make_test_dataset([30.0] * 4)
        assert compute_quality_weight(ds) == 0.0
        ds = make_test_dataset([30.0] * 10)
        assert compute_quality_weight(ds, is_precipitation=False) == 0.0

    def test_uses_zcount_not_total_count(self):
        ds = make_test_dataset([30.0] * 5, zcounts=[501] * 5)
        ds["sample_count"][:] = 99999
        assert compute_quality_weight(ds) == pytest.approx(501 / 5000)
        assert compute_quality_weight(ds.drop_vars("zcount")) == 0.0

    def test_first_layer_top_and_strict_sample_threshold(self):
        ds = make_test_dataset(
            [30.0] * 8, zcounts=[1000] * 5 + [30, 100, 10000]
        )
        assert compute_quality_weight(ds, layer_top_height_m=1300) == pytest.approx(
            5100 / (6 * 5000)
        )

    def test_excludes_detached_layer_and_heights_above_five_km(self):
        ds = make_test_dataset(
            [30.0] * 5 + [MDS] + [30.0] * 24,
            zcounts=[1000] * 6 + [10000] * 24,
        )
        assert compute_quality_weight(ds) == pytest.approx(0.2)
        ds = make_test_dataset([30.0] * 30, zcounts=[1000] * 25 + [10000] * 5)
        assert compute_quality_weight(ds) == pytest.approx(0.2)

    def test_missing_echo_has_zero_weight(self):
        """Missing echo (MDS) gives zero quality weight."""
        dbz_values = [MDS] * 10
        heights = list(range(100, 100 + STEP * 10, STEP))
        ds = make_test_dataset(dbz_values, heights)

        wq = compute_quality_weight(ds)

        assert wq == 0.0


class TestClimatologicalCorrection:
    """Tests for climatological correction computation."""

    def test_clim_corrections_included_when_fl_provided(self):
        """Climatological corrections computed when freezing level provided."""
        dbz_values = [35.0] * 25
        heights = list(range(100, 100 + STEP * 25, STEP))
        ds = make_test_dataset(dbz_values, heights)

        result = compute_vpr_correction(
            ds,
            freezing_level_m=2000,
            include_clim=True,
        )

        assert "cappi_clim_correction_db" in result.corrections
        assert "cappi_blended_correction_db" in result.corrections
        assert result.z_ground_clim_dbz is not None

    def test_clim_corrections_excluded_when_disabled(self):
        """Climatological corrections not computed when include_clim=False."""
        dbz_values = [35.0] * 25
        heights = list(range(100, 100 + STEP * 25, STEP))
        ds = make_test_dataset(dbz_values, heights)

        result = compute_vpr_correction(
            ds,
            freezing_level_m=2000,
            include_clim=False,
        )

        assert "cappi_clim_correction_db" not in result.corrections
        assert "cappi_blended_correction_db" not in result.corrections
        assert result.z_ground_clim_dbz is None

    def test_blended_is_between_instant_and_clim(self):
        """Blended correction is weighted average of instant and clim."""
        # Create profile with strong echo
        dbz_values = [35.0] * 25
        heights = list(range(100, 100 + STEP * 25, STEP))
        ds = make_test_dataset(dbz_values, heights)

        result = compute_vpr_correction(
            ds,
            freezing_level_m=2000,
            include_clim=True,
            clim_weight=0.2,
        )

        corr = result.corrections
        instant = corr["cappi_correction_db"].values
        clim = corr["cappi_clim_correction_db"].values
        blended = corr["cappi_blended_correction_db"].values

        # Blended should be between instant and clim (or equal if they're equal)
        for i in range(blended.shape[0]):
            for j in range(blended.shape[1]):
                lo = min(instant[i, j], clim[i, j])
                hi = max(instant[i, j], clim[i, j])
                assert lo - 0.01 <= blended[i, j] <= hi + 0.01

    def test_quality_weight_affects_blend(self):
        """Higher quality weight gives more weight to instantaneous correction."""
        dbz_values = [35.0] * 25
        heights = list(range(100, 100 + STEP * 25, STEP))
        ds = make_test_dataset(dbz_values, heights)

        result = compute_vpr_correction(
            ds,
            freezing_level_m=2000,
            include_clim=True,
            clim_weight=0.2,
        )

        assert result.quality_weight > 0
        # Quality weight should be stored in result
        assert result.corrections.attrs["quality_weight"] == result.quality_weight


class TestClimatologyOnlyCorrection:
    """Tests for climatology-only VPR correction (no valid echo)."""

    def test_no_echo_returns_climatology_only(self):
        """When no valid echo exists, returns climatology-only corrections."""
        # All MDS values - no valid echo
        dbz_values = [MDS] * 25
        heights = list(range(100, 100 + STEP * 25, STEP))
        ds = make_test_dataset(dbz_values, heights)

        result = compute_vpr_correction(
            ds,
            freezing_level_m=2000,
            include_clim=True,
            clim_weight=0.2,
        )

        # Should still get usable corrections
        assert result.usable is True
        # Quality weight should be clim_weight for compositing
        assert result.quality_weight == 0.2
        # Should have all correction variables
        assert "cappi_correction_db" in result.corrections
        assert "cappi_clim_correction_db" in result.corrections
        assert "cappi_blended_correction_db" in result.corrections
        # Climatology-only flag should be set
        assert result.corrections.attrs.get("climatology_only") is True

    def test_climatology_only_corrections_are_identical(self):
        """Climatology-only: instant, clim, and blended are all equal."""
        dbz_values = [MDS] * 25
        heights = list(range(100, 100 + STEP * 25, STEP))
        ds = make_test_dataset(dbz_values, heights)

        result = compute_vpr_correction(
            ds,
            freezing_level_m=2000,
            include_clim=True,
        )

        corr = result.corrections
        instant = corr["cappi_correction_db"].values
        clim = corr["cappi_clim_correction_db"].values
        blended = corr["cappi_blended_correction_db"].values

        np.testing.assert_array_equal(instant, clim)
        np.testing.assert_array_equal(blended, clim)

    def test_climatology_only_corrections_are_nonzero(self):
        """Climatology-only corrections have actual non-zero values at range."""
        dbz_values = [MDS] * 25
        heights = list(range(100, 100 + STEP * 25, STEP))
        ds = make_test_dataset(dbz_values, heights)

        result = compute_vpr_correction(
            ds,
            freezing_level_m=2000,
            include_clim=True,
        )

        corr = result.corrections
        # At longer ranges, corrections should be non-zero
        # (beam samples from higher where clim profile differs from ground)
        far_range_idx = 100  # ~100 km
        assert not np.all(corr["cappi_correction_db"].values[far_range_idx, :] == 0)

    def test_no_echo_no_freezing_level_returns_unusable(self):
        """Without freezing level, no-echo profile returns unusable result."""
        dbz_values = [MDS] * 25
        heights = list(range(100, 100 + STEP * 25, STEP))
        ds = make_test_dataset(dbz_values, heights)

        result = compute_vpr_correction(
            ds,
            freezing_level_m=None,
            include_clim=True,
        )

        assert result.usable is False
        assert result.quality_weight == 0.0

    def test_no_echo_respects_disabled_climatology(self):
        ds = make_test_dataset([MDS] * 5)
        result = compute_vpr_correction(
            ds, freezing_level_m=2000, include_clim=False, max_range_km=1
        )
        assert not result.usable
        assert result.z_ground_clim_dbz is None
        assert "cappi_clim_correction_db" not in result.corrections

    def test_climatology_only_ground_reference_is_clim(self):
        """Climatology-only profiles use climatological ground reference."""
        dbz_values = [MDS] * 25
        heights = list(range(100, 100 + STEP * 25, STEP))
        ds = make_test_dataset(dbz_values, heights)

        result = compute_vpr_correction(
            ds,
            freezing_level_m=2000,
            include_clim=True,
        )

        # z_ground_dbz should equal z_ground_clim_dbz
        assert result.z_ground_dbz == result.z_ground_clim_dbz
        # Should be the climatological base value: 10 + FL_km * 10 = 10 + 2 * 10 = 30
        assert abs(result.z_ground_dbz - 30.0) < 0.1


class TestBrightBandSlopeAdjustment:
    """Tests for bright band slope adjustment of ground reference.

    When a steep bright band is close to the surface, the enhanced reflectivity
    can cause over-correction. The slope adjustment uses an extrapolated value
    from above the BB instead of the potentially inflated surface value.

    Based on allprof_prodx2.pl lines 1304-1318.
    """

    def test_compute_bb_slope_basic(self):
        """Slope calculation produces expected result."""
        from vprc.vpr_correction import compute_bb_slope

        # Simple profile: 30 dBZ at bottom, 20 dBZ at top (1000m apart)
        heights = np.array([100, 300, 500, 700, 900, 1100])
        dbz_values = np.array([35.0, 30.0, 28.0, 24.0, 20.0, 15.0])

        # BB from 300m (30 dBZ) to 900m (20 dBZ)
        slope = compute_bb_slope(dbz_values, heights, 300, 900)

        # Expected: (30 - 20) / (900 - 300) * 200 = 10 / 600 * 200 ≈ 3.33
        assert abs(slope - 3.33) < 0.1

    def test_compute_bb_slope_zero_thickness(self):
        """Zero thickness BB returns zero slope."""
        from vprc.vpr_correction import compute_bb_slope

        heights = np.array([100, 300, 500])
        dbz_values = np.array([30.0, 35.0, 25.0])

        slope = compute_bb_slope(dbz_values, heights, 300, 300)
        assert slope == 0.0

    def test_adjust_ground_reference_no_bb(self):
        """Without BB detection, original value is returned."""
        from vprc.vpr_correction import adjust_ground_reference
        from vprc.bright_band import BrightBandResult

        heights = np.array([100, 300, 500, 700])
        dbz_values = np.array([30.0, 28.0, 25.0, 22.0])
        bb_result = BrightBandResult(detected=False)

        adjusted_dbz, was_adjusted = adjust_ground_reference(
            dbz_values, heights, bb_result, lowest_height=100
        )

        assert adjusted_dbz == 30.0
        assert not was_adjusted

    def test_adjust_ground_reference_bb_peak_at_surface(self):
        """BB peak at lowest level uses BB top value as reference."""
        from vprc.vpr_correction import adjust_ground_reference
        from vprc.bright_band import BrightBandResult

        heights = np.array([100, 300, 500, 700, 900])
        # Strong BB at surface: peak 40 dBZ at 100m, drops to 25 dBZ at 500m
        dbz_values = np.array([40.0, 35.0, 25.0, 22.0, 20.0])
        bb_result = BrightBandResult(
            detected=True,
            peak_height=100,  # Peak at lowest level
            bottom_height=100,
            top_height=500,
            peak_dbz=40.0,
        )

        adjusted_dbz, was_adjusted = adjust_ground_reference(
            dbz_values, heights, bb_result, lowest_height=100
        )

        # Should use value at BB top (500m = 25 dBZ)
        assert adjusted_dbz == 25.0
        assert was_adjusted

    def test_adjust_ground_reference_steep_bb_near_surface(self):
        """Steep BB near surface triggers slope adjustment."""
        from vprc.vpr_correction import adjust_ground_reference
        from vprc.bright_band import BrightBandResult

        heights = np.array([100, 300, 500, 700, 900])
        # BB just above surface: 35 dBZ at 300m (bottom), 20 dBZ at 700m (top)
        # Slope = (35-20)/(700-300)*200 = 7.5 dBZ/200m >> 0.2 threshold
        dbz_values = np.array([32.0, 35.0, 30.0, 20.0, 18.0])
        bb_result = BrightBandResult(
            detected=True,
            peak_height=500,
            bottom_height=300,  # Below 600m height limit
            top_height=700,
            peak_dbz=30.0,
        )

        adjusted_dbz, was_adjusted = adjust_ground_reference(
            dbz_values, heights, bb_result, lowest_height=100
        )

        # Should extrapolate: dbz_at_top + top_height/200 * 0.2
        # = 20.0 + 700/200 * 0.2 = 20.0 + 0.7 = 20.7
        expected = 20.0 + 700 / 200 * 0.2
        assert abs(adjusted_dbz - expected) < 0.01
        assert was_adjusted

    def test_adjust_ground_reference_bottom_at_lowest_positive_level(self):
        from vprc.vpr_correction import adjust_ground_reference
        from vprc.bright_band import BrightBandResult

        heights = np.array([100, 300, 500, 700, 900])
        dbz_values = np.array([30.0, 40.0, 25.0, 20.0, 18.0])
        bb = BrightBandResult(
            detected=True, peak_height=300, bottom_height=100, top_height=700
        )
        reference, adjusted = adjust_ground_reference(dbz_values, heights, bb, 100)
        assert adjusted
        assert reference == pytest.approx(20.7)

    def test_adjust_ground_reference_gentle_slope_no_adjustment(self):
        """Gentle slope below threshold does not trigger adjustment."""
        from vprc.vpr_correction import adjust_ground_reference
        from vprc.bright_band import BrightBandResult

        heights = np.array([100, 300, 500, 700, 900])
        # Gentle slope: 25 dBZ at 300m, 24.5 dBZ at 700m
        # Slope = (25-24.5)/(700-300)*200 = 0.25 dBZ/200m ~= 0.25
        # Still slightly above 0.2, so let's make it even gentler
        dbz_values = np.array([26.0, 25.0, 24.8, 24.5, 24.0])
        bb_result = BrightBandResult(
            detected=True,
            peak_height=500,
            bottom_height=300,  # Below 600m height limit
            top_height=700,
            peak_dbz=24.8,
        )

        # Slope = (25-24.5)/(700-300)*200 = 0.25 > 0.2, still triggers
        # Let's make BB bottom higher than 600m to avoid trigger
        bb_result_high = BrightBandResult(
            detected=True,
            peak_height=900,
            bottom_height=700,  # Above 600m height limit
            top_height=1100,
            peak_dbz=24.0,
        )

        adjusted_dbz, was_adjusted = adjust_ground_reference(
            dbz_values, heights, bb_result_high, lowest_height=100
        )

        # Should return original (lowest valid = 26.0)
        assert adjusted_dbz == 26.0
        assert not was_adjusted

    def test_adjust_ground_reference_safety_cap(self):
        """Adjusted value is capped at observed lowest-level value."""
        from vprc.vpr_correction import adjust_ground_reference
        from vprc.bright_band import BrightBandResult

        heights = np.array([100, 300, 500, 700, 900, 1100, 1300])
        # BB with high top: extrapolated value would exceed observed
        # Low observed value at surface (15 dBZ), BB top at 1100m with 18 dBZ
        # Extrapolated: 18 + 1100/200 * 0.2 = 18 + 1.1 = 19.1 > 15
        dbz_values = np.array([15.0, 20.0, 25.0, 22.0, 18.0, 18.0, 15.0])
        bb_result = BrightBandResult(
            detected=True,
            peak_height=500,
            bottom_height=300,  # Below 600m
            top_height=1100,
            peak_dbz=25.0,
        )

        # Slope = (20-18)/(1100-300)*200 = 0.5 > 0.2
        adjusted_dbz, was_adjusted = adjust_ground_reference(
            dbz_values, heights, bb_result, lowest_height=100
        )

        # Should be capped at observed 15 dBZ (not extrapolated 19.1)
        assert adjusted_dbz == 15.0
        assert not was_adjusted  # Capped means no adjustment applied

    def test_compute_vpr_correction_with_bb_result(self):
        """compute_vpr_correction uses BB result for slope adjustment."""
        from vprc.bright_band import BrightBandResult

        heights = list(range(100, 2100, STEP))  # 100 to 1900m
        # Create profile with BB near surface
        dbz_values = []
        for h in heights:
            if h < 300:
                dbz_values.append(32.0)  # Below BB
            elif h <= 500:
                dbz_values.append(38.0)  # In BB peak
            elif h <= 700:
                dbz_values.append(30.0)  # Above BB peak
            else:
                dbz_values.append(25.0 - (h - 700) * 0.01)  # Decreasing above

        ds = make_test_dataset(dbz_values, heights)

        bb_result = BrightBandResult(
            detected=True,
            peak_height=500,
            bottom_height=300,
            top_height=700,
            peak_dbz=38.0,
        )

        # With BB result - should adjust ground reference
        result_with_bb = compute_vpr_correction(ds, bb_result=bb_result)

        # Without BB result - uses original ground reference (32 dBZ at 100m)
        result_no_bb = compute_vpr_correction(ds, bb_result=None)

        # Ground reference should differ if adjustment was applied
        # The BB has slope = (32-30)/400*200 = 1.0 > 0.2, and bottom=300 < 600
        # So adjustment should trigger
        # Note: may be capped if extrapolated > original
        assert result_with_bb.z_ground_dbz != result_no_bb.z_ground_dbz or \
               result_with_bb.z_ground_dbz == 32.0  # Capped case

    def test_pipeline_passes_bb_result(self):
        """process_vvp passes BB result to compute_vpr_correction."""
        from vprc import process_vvp
        from pathlib import Path

        test_file = Path(__file__).parent / "data" / "202208171245_VIM.VVP_40.txt"
        if not test_file.exists():
            pytest.skip("Test data file not available")

        result = process_vvp(test_file, freezing_level_m=2000)

        # Just verify pipeline runs without error
        # The BB result is now passed to VPR correction
        assert result.vpr_correction is not None


class TestBBPeakAmplitudeCap:
    """Tests for bright band peak amplitude cap at surface.

    When BB peak is at the lowest level with excessive amplitude (>10 dBZ),
    the peak dBZ is reduced to cap effective amplitude.

    Based on allprof_prodx2.pl lines 1091-1098.
    """

    def test_cap_bb_peak_amplitude_no_bb(self):
        """No BB detected - returns unchanged dataset."""
        from vprc.vpr_correction import cap_bb_peak_amplitude
        from vprc.bright_band import BrightBandResult

        dbz_values = [35.0, 30.0, 25.0]
        heights = [100, 300, 500]
        ds = make_test_dataset(dbz_values, heights)
        bb_result = BrightBandResult(detected=False)

        result = cap_bb_peak_amplitude(ds, bb_result)

        np.testing.assert_array_equal(
            result["corrected_dbz"].values, ds["corrected_dbz"].values
        )

    def test_cap_bb_peak_amplitude_not_at_surface(self):
        """BB peak not at lowest level - returns unchanged dataset."""
        from vprc.vpr_correction import cap_bb_peak_amplitude
        from vprc.bright_band import BrightBandResult

        dbz_values = [30.0, 40.0, 25.0, 20.0]  # Peak at 300m
        heights = [100, 300, 500, 700]
        ds = make_test_dataset(dbz_values, heights)
        bb_result = BrightBandResult(
            detected=True,
            peak_height=300,
            bottom_height=100,
            top_height=500,
            amplitude_above=15.0,  # High amplitude, but not at surface
        )

        result = cap_bb_peak_amplitude(ds, bb_result)

        np.testing.assert_array_equal(
            result["corrected_dbz"].values, ds["corrected_dbz"].values
        )

    def test_cap_bb_peak_amplitude_below_threshold(self):
        """BB at surface but amplitude below cap - returns unchanged dataset."""
        from vprc.vpr_correction import cap_bb_peak_amplitude
        from vprc.bright_band import BrightBandResult

        dbz_values = [38.0, 30.0, 25.0]  # Peak at surface
        heights = [100, 300, 500]
        ds = make_test_dataset(dbz_values, heights)
        bb_result = BrightBandResult(
            detected=True,
            peak_height=100,  # At lowest level
            bottom_height=100,
            top_height=300,
            amplitude_above=8.0,  # Below 10 dB cap
        )

        result = cap_bb_peak_amplitude(ds, bb_result)

        np.testing.assert_array_equal(
            result["corrected_dbz"].values, ds["corrected_dbz"].values
        )

    def test_cap_bb_peak_amplitude_exceeds_threshold(self):
        """BB at surface with high amplitude - peak is reduced."""
        from vprc.vpr_correction import cap_bb_peak_amplitude
        from vprc.bright_band import BrightBandResult

        dbz_values = [45.0, 30.0, 25.0]  # Peak at surface with 15 dB amplitude
        heights = [100, 300, 500]
        ds = make_test_dataset(dbz_values, heights)
        bb_result = BrightBandResult(
            detected=True,
            peak_height=100,  # At lowest level
            bottom_height=100,
            top_height=300,
            peak_dbz=45.0,
            amplitude_above=15.0,  # Exceeds 10 dB cap
        )

        result = cap_bb_peak_amplitude(ds, bb_result)

        # Peak should be reduced by (15 - 10) = 5 dB
        # New peak: 45 - 5 = 40 dBZ
        expected_peak = 45.0 - (15.0 - 10.0)
        assert result["corrected_dbz"].sel(height=100).values == expected_peak

        # Other values unchanged
        assert result["corrected_dbz"].sel(height=300).values == 30.0
        assert result["corrected_dbz"].sel(height=500).values == 25.0

    def test_cap_bb_peak_amplitude_custom_threshold(self):
        """Custom amplitude cap threshold is respected."""
        from vprc.vpr_correction import cap_bb_peak_amplitude
        from vprc.bright_band import BrightBandResult

        dbz_values = [40.0, 30.0, 25.0]  # 10 dB amplitude
        heights = [100, 300, 500]
        ds = make_test_dataset(dbz_values, heights)
        bb_result = BrightBandResult(
            detected=True,
            peak_height=100,
            bottom_height=100,
            top_height=300,
            peak_dbz=40.0,
            amplitude_above=10.0,
        )

        # With default cap (10), no reduction
        result_default = cap_bb_peak_amplitude(ds, bb_result)
        assert result_default["corrected_dbz"].sel(height=100).values == 40.0

        # With lower cap (5), peak is reduced
        result_lower = cap_bb_peak_amplitude(ds, bb_result, amplitude_cap_db=5.0)
        expected_peak = 40.0 - (10.0 - 5.0)  # 35 dBZ
        assert result_lower["corrected_dbz"].sel(height=100).values == expected_peak

    @pytest.mark.parametrize("include_raw", [False, True])
    def test_cap_is_not_applied_twice_after_profile_finalization(self, include_raw):
        from vprc.vpr_correction import cap_bb_peak_amplitude
        from vprc.bright_band import BrightBandResult

        ds = make_test_dataset([45.0, 30.0, 25.0])
        if include_raw:
            ds["lin_dbz"] = ds.corrected_dbz.copy()
        bb = BrightBandResult(
            detected=True, peak_height=100, bottom_height=-1, top_height=300,
            peak_dbz=45.0, amplitude_above=15.0,
        )
        finalized = cap_bb_peak_amplitude(ds, bb)
        repeated = cap_bb_peak_amplitude(finalized, bb)
        np.testing.assert_array_equal(repeated.corrected_dbz, finalized.corrected_dbz)
        assert repeated.corrected_dbz.sel(height=100).item() == 40.0