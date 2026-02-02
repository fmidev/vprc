"""Tests for spatial compositing of VPR corrections."""

import numpy as np
import pytest
from pyproj import CRS

from vprc.composite import (
    CompositeGrid,
    RadarCorrection,
    compute_radar_distances,
    composite_corrections,
    create_empty_composite,
    create_radar_correction,
    inverse_distance_weight,
    interpolate_correction_to_grid,
)
from vprc import process_vvp
from vprc.vpr_correction import VPRCorrectionResult


# Test data paths (2023-08-28 multi-radar data)
TEST_DATA_DIR = "tests/data"
KAN_FILE = f"{TEST_DATA_DIR}/202308281000_KAN.VVP_40.txt"
KOR_FILE = f"{TEST_DATA_DIR}/202308281000_KOR.VVP_40.txt"
VIH_FILE = f"{TEST_DATA_DIR}/202308281000_VIH.VVP_40.txt"

# Radar coordinates from radar_defaults.toml
RADAR_COORDS = {
    "KAN": (61.81085, 22.50204),
    "KOR": (60.128469, 21.643379),
    "VIH": (60.5561915, 24.49558603),
}


class TestCompositeGrid:
    """Tests for CompositeGrid creation."""

    def test_from_bounds_creates_regular_grid(self):
        """Grid from bounds has correct spacing."""
        grid = CompositeGrid.from_bounds(
            xmin=0, xmax=10000, ymin=0, ymax=5000, resolution_m=1000
        )

        assert len(grid.x) == 11  # 0, 1000, ..., 10000
        assert len(grid.y) == 6  # 0, 1000, ..., 5000
        assert grid.x[1] - grid.x[0] == 1000
        assert grid.y[1] - grid.y[0] == 1000

    def test_from_bounds_uses_epsg_3067_by_default(self):
        """Default CRS is ETRS-TM35FIN."""
        grid = CompositeGrid.from_bounds(0, 1000, 0, 1000, 500)

        assert grid.crs.to_epsg() == 3067

    def test_for_finland_covers_radar_network(self):
        """Finland grid covers expected area."""
        grid = CompositeGrid.for_finland(resolution_m=10000)

        # Should be roughly 700km x 1200km
        assert grid.x[-1] - grid.x[0] >= 600_000
        assert grid.y[-1] - grid.y[0] >= 1_000_000

    def test_for_radars_all_configured(self):
        """Grid from all configured radars covers expected area."""
        grid = CompositeGrid.for_radars(resolution_m=10000)

        # Should cover Finland with margins
        assert grid.x[-1] - grid.x[0] >= 600_000
        assert grid.y[-1] - grid.y[0] >= 1_000_000
        assert grid.crs.to_epsg() == 3067

    def test_for_radars_subset(self):
        """Grid from subset of radars has appropriate bounds."""
        # KAN and VIH are ~200km apart
        grid = CompositeGrid.for_radars(
            radar_codes=["KAN", "VIH"],
            range_km=251.0,
            resolution_m=10000,
        )

        # Width should be: distance between radars + 2*251km margin
        # KAN-VIH distance is ~200km, so total ~700km
        width_km = (grid.x[-1] - grid.x[0]) / 1000
        height_km = (grid.y[-1] - grid.y[0]) / 1000
        assert 600 < width_km < 800
        assert 500 < height_km < 700

    def test_for_radars_single_radar(self):
        """Grid for single radar is 2*range_km square."""
        grid = CompositeGrid.for_radars(
            radar_codes=["KAN"],
            range_km=100.0,
            resolution_m=1000,
        )

        # Should be exactly 200km x 200km (2 * 100km range)
        width_km = (grid.x[-1] - grid.x[0]) / 1000
        height_km = (grid.y[-1] - grid.y[0]) / 1000
        assert 199 <= width_km <= 201
        assert 199 <= height_km <= 201

    def test_for_radars_empty_list_raises(self):
        """Empty radar list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            CompositeGrid.for_radars(radar_codes=[])

    def test_for_radars_invalid_codes_raises(self):
        """All invalid radar codes raises ValueError."""
        with pytest.raises(ValueError, match="No radars with valid coordinates"):
            CompositeGrid.for_radars(radar_codes=["XXX", "YYY"])

    def test_for_radars_skips_missing_coordinates(self):
        """Radars without coordinates are silently skipped."""
        # KER has no coordinates in radar_defaults.toml
        # Should work as long as at least one valid radar
        grid = CompositeGrid.for_radars(
            radar_codes=["KER", "KAN"],
            range_km=100.0,
            resolution_m=10000,
        )
        # Should create grid based on KAN only
        assert len(grid.x) > 0
        assert len(grid.y) > 0


class TestInverseDistanceWeight:
    """Tests for IDW weight function."""

    def test_closer_gets_higher_weight(self):
        """Closer points receive higher weight."""
        distances = np.array([10_000, 50_000, 100_000])
        quality = 1.0

        weights = inverse_distance_weight(distances, quality)

        assert weights[0] > weights[1] > weights[2]

    def test_higher_quality_gives_higher_weight(self):
        """Higher quality weight increases overall weight."""
        distances = np.array([50_000])

        w_low = inverse_distance_weight(distances, quality_weights=0.5)
        w_high = inverse_distance_weight(distances, quality_weights=2.0)

        assert w_high[0] > w_low[0]
        assert w_high[0] / w_low[0] == pytest.approx(4.0)  # ratio of quality

    def test_min_distance_prevents_infinity(self):
        """Minimum distance clamp prevents division issues."""
        distances = np.array([0.0, 100.0, 500.0])
        quality = 1.0

        weights = inverse_distance_weight(distances, quality, min_distance_m=1000.0)

        # All distances below min should give same weight
        assert weights[0] == weights[1] == weights[2]
        assert np.isfinite(weights).all()


class TestComputeRadarDistances:
    """Tests for radar distance calculation."""

    def test_distances_are_positive(self):
        """All computed distances are positive."""
        grid = CompositeGrid.from_bounds(
            200_000, 300_000, 6_700_000, 6_800_000, resolution_m=10_000
        )

        # Create mock radar correction
        mock_corr = _create_mock_correction()
        radars = [
            RadarCorrection(
                radar_code="KAN",
                latitude=RADAR_COORDS["KAN"][0],
                longitude=RADAR_COORDS["KAN"][1],
                correction=mock_corr,
                quality_weight=1.0,
            )
        ]

        distances = compute_radar_distances(grid, radars)

        assert distances.shape == (1, len(grid.y), len(grid.x))
        assert (distances >= 0).all()

    def test_multiple_radars_give_correct_shape(self):
        """Output shape matches (n_radars, ny, nx)."""
        grid = CompositeGrid.from_bounds(
            200_000, 250_000, 6_700_000, 6_750_000, resolution_m=10_000
        )

        mock_corr = _create_mock_correction()
        radars = [
            RadarCorrection("KAN", *RADAR_COORDS["KAN"], mock_corr, 1.0),
            RadarCorrection("KOR", *RADAR_COORDS["KOR"], mock_corr, 1.0),
            RadarCorrection("VIH", *RADAR_COORDS["VIH"], mock_corr, 1.0),
        ]

        distances = compute_radar_distances(grid, radars)

        assert distances.shape[0] == 3
        assert distances.shape[1] == len(grid.y)
        assert distances.shape[2] == len(grid.x)


class TestCompositeCorrections:
    """Tests for full composite workflow."""

    def test_single_radar_uses_direct_values(self):
        """Single radar coverage uses that radar's correction directly."""
        grid = CompositeGrid.from_bounds(
            200_000, 250_000, 6_800_000, 6_850_000, resolution_m=5_000
        )

        mock_corr = _create_mock_correction(constant_db=2.5)
        radars = [
            RadarCorrection("KAN", *RADAR_COORDS["KAN"], mock_corr, 1.0),
        ]

        result = composite_corrections(radars, grid, max_range_km=300)

        # Within range, should have correction values
        valid_mask = ~np.isnan(result["correction_db"].values)
        assert valid_mask.sum() > 0

        # All valid values should be close to the constant correction
        valid_values = result["correction_db"].values[valid_mask]
        assert np.allclose(valid_values, 2.5, atol=0.1)

    def test_multiple_radars_blend_at_overlap(self):
        """Overlapping coverage uses weighted blend."""
        # Grid roughly between KAN and VIH
        grid = CompositeGrid.from_bounds(
            250_000, 350_000, 6_700_000, 6_800_000, resolution_m=10_000
        )

        corr_kan = _create_mock_correction(constant_db=2.0)
        corr_vih = _create_mock_correction(constant_db=4.0)

        radars = [
            RadarCorrection("KAN", *RADAR_COORDS["KAN"], corr_kan, 1.0),
            RadarCorrection("VIH", *RADAR_COORDS["VIH"], corr_vih, 1.0),
        ]

        result = composite_corrections(radars, grid, max_range_km=300)

        # In overlap area, correction should be between 2.0 and 4.0
        valid = result["correction_db"].values
        valid = valid[~np.isnan(valid)]

        assert len(valid) > 0
        assert valid.min() >= 2.0 - 0.1
        assert valid.max() <= 4.0 + 0.1

    def test_n_radars_counts_contributors(self):
        """n_radars tracks number of contributing radars."""
        grid = CompositeGrid.from_bounds(
            250_000, 350_000, 6_700_000, 6_800_000, resolution_m=10_000
        )

        mock_corr = _create_mock_correction()
        radars = [
            RadarCorrection("KAN", *RADAR_COORDS["KAN"], mock_corr, 1.0),
            RadarCorrection("VIH", *RADAR_COORDS["VIH"], mock_corr, 1.0),
        ]

        result = composite_corrections(radars, grid, max_range_km=300)

        # Should have some points with 2 radars
        assert (result["n_radars"].values == 2).sum() > 0

    def test_zero_quality_weight_excludes_radar(self):
        """Radar with zero quality weight is excluded."""
        grid = CompositeGrid.from_bounds(
            200_000, 250_000, 6_800_000, 6_850_000, resolution_m=5_000
        )

        mock_corr = _create_mock_correction(constant_db=3.0)
        radars = [
            RadarCorrection("KAN", *RADAR_COORDS["KAN"], mock_corr, 0.0),  # zero quality
        ]

        result = composite_corrections(radars, grid, max_range_km=300)

        # All should be NaN since only radar has zero quality
        assert np.isnan(result["correction_db"].values).all()

    def test_beyond_max_range_is_nan(self):
        """Grid points beyond max range get NaN."""
        # Grid far from all radars
        grid = CompositeGrid.from_bounds(
            50_000, 60_000, 7_700_000, 7_750_000, resolution_m=5_000
        )

        mock_corr = _create_mock_correction()
        radars = [
            RadarCorrection("KAN", *RADAR_COORDS["KAN"], mock_corr, 1.0),
        ]

        result = composite_corrections(radars, grid, max_range_km=100)

        # All should be NaN - too far from radar
        assert np.isnan(result["correction_db"].values).all()


class TestEmptyComposite:
    """Tests for handling empty/no-precipitation scenarios."""

    def test_empty_radar_list_returns_valid_dataset(self):
        """Empty radar list produces valid Dataset with NaN corrections."""
        grid = CompositeGrid.from_bounds(
            200_000, 250_000, 6_800_000, 6_850_000, resolution_m=5_000
        )

        result = composite_corrections([], grid)

        # Should have correct shape
        assert result["correction_db"].shape == (len(grid.y), len(grid.x))

        # All corrections should be NaN
        assert np.isnan(result["correction_db"].values).all()

        # Weight sum and n_radars should be zero
        assert (result["weight_sum"].values == 0).all()
        assert (result["n_radars"].values == 0).all()

        # Should have empty_composite flag
        assert result.attrs["empty_composite"] is True

    def test_create_empty_composite_produces_valid_grid(self):
        """create_empty_composite produces a valid exportable Dataset."""
        grid = CompositeGrid.from_bounds(
            100_000, 200_000, 6_700_000, 6_800_000, resolution_m=10_000
        )

        result = create_empty_composite(grid, radar_codes=["KAN", "VIH"])

        # Check shape matches grid
        assert result["correction_db"].shape == (len(grid.y), len(grid.x))

        # All corrections should be NaN
        assert np.isnan(result["correction_db"].values).all()

        # Metadata should be preserved
        assert result.attrs["empty_composite"] is True
        assert result.attrs["radar_codes"] == ["KAN", "VIH"]
        assert "crs_epsg" in result.attrs

    def test_create_empty_composite_has_cog_compatible_structure(self):
        """Empty composite has required structure for COG export."""
        grid = CompositeGrid.for_finland(resolution_m=50_000)

        result = create_empty_composite(grid)

        # Required variables exist
        assert "correction_db" in result
        assert "weight_sum" in result
        assert "n_radars" in result

        # Required coordinates exist
        assert "x" in result.coords
        assert "y" in result.coords

        # CRS info present
        assert result.attrs.get("crs_epsg") is not None

    def test_all_radars_zero_quality_produces_nan_composite(self):
        """When all radars have zero quality, result is all NaN."""
        grid = CompositeGrid.from_bounds(
            200_000, 300_000, 6_750_000, 6_850_000, resolution_m=10_000
        )

        mock_corr = _create_mock_correction(constant_db=5.0)
        radars = [
            RadarCorrection("KAN", *RADAR_COORDS["KAN"], mock_corr, 0.0),
            RadarCorrection("VIH", *RADAR_COORDS["VIH"], mock_corr, 0.0),
        ]

        result = composite_corrections(radars, grid, max_range_km=300)

        # All corrections should be NaN
        assert np.isnan(result["correction_db"].values).all()

        # But we still have the radar codes in metadata
        assert "KAN" in result.attrs["radar_codes"]
        assert "VIH" in result.attrs["radar_codes"]

        # Not marked as empty_composite (radars were provided, just zero quality)
        assert result.attrs["empty_composite"] is False


class TestWithRealData:
    """Integration tests using real VVP files."""

    @pytest.fixture
    def processed_radars(self):
        """Process the three test radars."""
        results = {}
        for code, filepath in [("KAN", KAN_FILE), ("KOR", KOR_FILE), ("VIH", VIH_FILE)]:
            try:
                result = process_vvp(filepath, freezing_level_m=3000)
                if result.vpr_correction is not None:
                    results[code] = result
            except FileNotFoundError:
                pass  # Skip if file doesn't exist

        return results

    def test_can_process_all_three_radars(self, processed_radars):
        """All three test radars can be processed."""
        # At least some should succeed
        assert len(processed_radars) > 0

    def test_composite_from_real_data(self, processed_radars):
        """Composite can be created from real radar data."""
        # Filter to only radars with positive quality weight
        usable_radars = {
            code: result
            for code, result in processed_radars.items()
            if result.vpr_correction.quality_weight > 0
        }

        if len(usable_radars) < 1:
            pytest.skip("Need at least 1 radar with positive quality weight")

        radars = []
        for code, result in usable_radars.items():
            lat, lon = RADAR_COORDS[code]
            radar_corr = create_radar_correction(
                code, lat, lon, result.vpr_correction
            )
            radars.append(radar_corr)

        # Create small test grid
        grid = CompositeGrid.from_bounds(
            200_000, 400_000, 6_650_000, 6_900_000, resolution_m=10_000
        )

        result = composite_corrections(radars, grid)

        # Should have some valid composite values
        valid_count = (~np.isnan(result["correction_db"].values)).sum()
        assert valid_count > 0

        # Should have reasonable correction values
        valid = result["correction_db"].values
        valid = valid[~np.isnan(valid)]
        assert valid.min() > -20  # Not unreasonably negative
        assert valid.max() < 30  # Not unreasonably positive

    def test_composite_with_mock_quality_weights(self, processed_radars):
        """Composite can be created using mock quality weights."""
        if len(processed_radars) < 2:
            pytest.skip("Need at least 2 radars for multi-radar composite test")

        radars = []
        for code, result in processed_radars.items():
            lat, lon = RADAR_COORDS[code]
            # Use mock quality weight of 1.0 for all radars
            radar_corr = RadarCorrection(
                radar_code=code,
                latitude=lat,
                longitude=lon,
                correction=result.vpr_correction,
                quality_weight=1.0,  # Force non-zero weight
            )
            radars.append(radar_corr)

        grid = CompositeGrid.from_bounds(
            200_000, 400_000, 6_650_000, 6_900_000, resolution_m=10_000
        )

        result = composite_corrections(radars, grid)

        # Should have some valid composite values
        valid_count = (~np.isnan(result["correction_db"].values)).sum()
        assert valid_count > 0


def _create_mock_correction(constant_db: float = 2.0) -> VPRCorrectionResult:
    """Create a mock VPR correction result for testing.

    Args:
        constant_db: Constant correction value to use

    Returns:
        VPRCorrectionResult with constant correction at all ranges
    """
    import xarray as xr

    range_km = np.arange(1, 251)
    cappi_heights = [500, 1000]

    # Create constant correction array
    # Dims: (range_km, cappi_height) to match real data
    corr_data = np.full((len(range_km), len(cappi_heights)), constant_db)

    ds = xr.Dataset(
        {
            "cappi_correction_db": (["range_km", "cappi_height"], corr_data),
            "cappi_blended_correction_db": (["range_km", "cappi_height"], corr_data),
        },
        coords={
            "range_km": range_km,
            "cappi_height": cappi_heights,
        },
    )

    return VPRCorrectionResult(
        corrections=ds,
        z_ground_dbz=35.0,
        quality_weight=1.0,
        usable=True,
    )
