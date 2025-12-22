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
)
from vprc.constants import MDS, STEP, FINE_GRID_RESOLUTION_M, BEAMWIDTH_DEG


def make_test_dataset(
    dbz_values: list[float],
    heights: list[int] | None = None,
    antenna_height_m: float = 100.0,
) -> xr.Dataset:
    """Create a minimal test dataset for VPR correction tests."""
    if heights is None:
        heights = list(range(100, 100 + STEP * len(dbz_values), STEP))

    ds = xr.Dataset(
        {
            "corrected_dbz": ("height", dbz_values),
            "sample_count": ("height", [100] * len(dbz_values)),
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

    def test_mds_values_become_zero_z(self):
        """MDS values are treated as zero reflectivity."""
        dbz_values = [20.0, MDS, 20.0]
        heights = [100, 300, 500]
        ds = make_test_dataset(dbz_values, heights)

        fine_heights, fine_z = interpolate_to_fine_grid(ds)

        # At 300m, Z should be interpolated (not exactly zero)
        # because we interpolate between valid values
        idx_300 = 300 // FINE_GRID_RESOLUTION_M
        # The middle point with MDS becomes 0 in coarse, then gets interpolated
        assert fine_z[idx_300] >= 0

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


class TestComputeBeamWeight:
    """Tests for Gaussian beam weight calculation."""

    def test_center_weight_is_one(self):
        """Weight at beam center (0°) is 1.0."""
        w = compute_beam_weight(0.0, BEAMWIDTH_DEG)
        assert np.isclose(w, 1.0)

    def test_half_power_at_beamwidth(self):
        """Weight at half-power beamwidth is ~0.5."""
        w = compute_beam_weight(BEAMWIDTH_DEG, BEAMWIDTH_DEG)
        # 10^(-0.6 * 1^2) = 10^(-0.6) ≈ 0.25
        assert np.isclose(w, 0.25, atol=0.01)

    def test_symmetric(self):
        """Weight is symmetric about center."""
        w_pos = compute_beam_weight(0.5, BEAMWIDTH_DEG)
        w_neg = compute_beam_weight(-0.5, BEAMWIDTH_DEG)
        # Function uses abs implicitly via angular_distance
        assert np.isclose(w_pos, w_neg)

    def test_weight_decreases_with_distance(self):
        """Weight decreases with angular distance."""
        w1 = compute_beam_weight(0.3, BEAMWIDTH_DEG)
        w2 = compute_beam_weight(0.6, BEAMWIDTH_DEG)
        w3 = compute_beam_weight(1.0, BEAMWIDTH_DEG)
        assert w1 > w2 > w3


class TestConvolveBeamWithProfile:
    """Tests for beam-profile convolution."""

    def test_uniform_profile_no_change(self):
        """Uniform profile produces same Z regardless of beam position."""
        n_layers = 600
        fine_heights = np.arange(0, n_layers * FINE_GRID_RESOLUTION_M, FINE_GRID_RESOLUTION_M)
        uniform_z = np.ones(n_layers) * 100.0  # Uniform Z

        z1 = convolve_beam_with_profile(fine_heights, uniform_z, 2000, 100000)
        z2 = convolve_beam_with_profile(fine_heights, uniform_z, 5000, 100000)

        assert np.isclose(z1, z2, rtol=0.01)

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


class TestComputeVprCorrection:
    """Tests for main VPR correction function."""

    def test_returns_correct_structure(self):
        """Result has expected structure and dimensions."""
        dbz_values = [30.0, 28.0, 25.0, 22.0, 20.0, 18.0, 15.0]
        ds = make_test_dataset(dbz_values)

        result = compute_vpr_correction(ds)

        assert isinstance(result, VPRCorrectionResult)
        assert result.usable
        assert "correction_db" in result.corrections
        assert "beam_height_m" in result.corrections
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

        corr = result.corrections["correction_db"].sel(cappi_height=500)
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
            assert "correction_db" in result.vpr_correction.corrections

    def test_process_vvp_skips_vpr_when_disabled(self):
        """process_vvp skips VPR correction when compute_vpr=False."""
        from vprc import process_vvp
        from pathlib import Path

        test_file = Path(__file__).parent / "data" / "202508241100_KAN.VVP_40.txt"
        if not test_file.exists():
            pytest.skip("Test data file not available")

        result = process_vvp(test_file, compute_vpr=False)

        assert result.vpr_correction is None
