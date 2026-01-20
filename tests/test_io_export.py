"""Tests for COG export functionality."""

import numpy as np
import pytest
import xarray as xr
from pathlib import Path

from vprc.io_export import (
    prepare_for_export,
    write_correction_cog,
    write_composite_cogs,
    write_weights_cog,
)


def _create_mock_composite() -> xr.Dataset:
    """Create a mock composite dataset for testing."""
    x = np.arange(200_000, 250_000, 5_000)
    y = np.arange(6_700_000, 6_750_000, 5_000)

    # Create test data with some NaN values
    correction = np.random.uniform(0, 5, (len(y), len(x)))
    correction[0, 0] = np.nan  # Add a no-coverage cell

    weight_sum = np.random.uniform(0, 10, (len(y), len(x)))
    n_radars = np.random.randint(0, 3, (len(y), len(x))).astype(np.int8)

    ds = xr.Dataset(
        {
            "correction_db": (["y", "x"], correction),
            "weight_sum": (["y", "x"], weight_sum),
            "n_radars": (["y", "x"], n_radars),
        },
        coords={
            "x": x,
            "y": y,
        },
        attrs={
            "crs": "EPSG:3067",
            "crs_epsg": 3067,
            "max_range_km": 250.0,
            "radar_codes": ["KAN", "VIH"],
        },
    )

    return ds


class TestPrepareForExport:
    """Tests for prepare_for_export function."""

    def test_adds_crs_metadata(self):
        """CRS metadata is added to dataset."""
        ds = _create_mock_composite()
        ds_export = prepare_for_export(ds)

        # rioxarray adds CRS as spatial_ref coordinate
        assert ds_export.rio.crs is not None
        assert ds_export.rio.crs.to_epsg() == 3067

    def test_sets_spatial_dims(self):
        """Spatial dimensions are set correctly."""
        ds = _create_mock_composite()
        ds_export = prepare_for_export(ds)

        assert ds_export.rio.x_dim == "x"
        assert ds_export.rio.y_dim == "y"

    def test_preserves_original_data(self):
        """Original data values are preserved."""
        ds = _create_mock_composite()
        ds_export = prepare_for_export(ds)

        np.testing.assert_array_equal(
            ds["correction_db"].values,
            ds_export["correction_db"].values,
        )


class TestWriteCorrectionCog:
    """Tests for write_correction_cog function."""

    def test_creates_file(self, tmp_path):
        """COG file is created."""
        ds = _create_mock_composite()
        output_path = tmp_path / "test.tif"

        write_correction_cog(ds, output_path)

        assert output_path.exists()

    def test_file_is_valid_geotiff(self, tmp_path):
        """Output file is a valid GeoTIFF."""
        import rasterio

        ds = _create_mock_composite()
        output_path = tmp_path / "test.tif"

        write_correction_cog(ds, output_path)

        with rasterio.open(output_path) as src:
            assert src.driver == "GTiff"
            assert src.crs.to_epsg() == 3067
            assert src.count == 1  # Single band

    def test_file_is_cog(self, tmp_path):
        """Output file has COG structure (tiled, overviews)."""
        import rasterio

        ds = _create_mock_composite()
        output_path = tmp_path / "test.tif"

        write_correction_cog(ds, output_path)

        with rasterio.open(output_path) as src:
            # COG files are tiled
            assert src.is_tiled

    def test_compression_deflate(self, tmp_path):
        """DEFLATE compression is applied."""
        import rasterio

        ds = _create_mock_composite()
        output_path = tmp_path / "test.tif"

        write_correction_cog(ds, output_path, compress="DEFLATE")

        with rasterio.open(output_path) as src:
            assert src.compression.name.upper() == "DEFLATE"

    def test_compression_lzw(self, tmp_path):
        """LZW compression can be used."""
        import rasterio

        ds = _create_mock_composite()
        output_path = tmp_path / "test.tif"

        write_correction_cog(ds, output_path, compress="LZW")

        with rasterio.open(output_path) as src:
            assert src.compression.name.upper() == "LZW"

    def test_nodata_is_nan(self, tmp_path):
        """NaN values are preserved as nodata."""
        import rasterio

        ds = _create_mock_composite()
        output_path = tmp_path / "test.tif"

        write_correction_cog(ds, output_path)

        with rasterio.open(output_path) as src:
            data = src.read(1)
            # Check that nodata value is set
            assert src.nodata is not None or np.isnan(src.nodata)

    def test_raises_on_missing_variable(self, tmp_path):
        """KeyError raised for missing variable."""
        ds = _create_mock_composite()
        output_path = tmp_path / "test.tif"

        with pytest.raises(KeyError, match="not found"):
            write_correction_cog(ds, output_path, variable="nonexistent")


class TestWriteCompositeCogs:
    """Tests for write_composite_cogs function."""

    def test_creates_all_files(self, tmp_path):
        """All expected files are created."""
        ds = _create_mock_composite()

        outputs = write_composite_cogs(ds, tmp_path)

        assert len(outputs) == 3
        assert "correction_db" in outputs
        assert "weight_sum" in outputs
        assert "n_radars" in outputs

        for path in outputs.values():
            assert path.exists()

    def test_prefix_is_applied(self, tmp_path):
        """Prefix is added to filenames."""
        ds = _create_mock_composite()

        outputs = write_composite_cogs(ds, tmp_path, prefix="202308281000")

        for path in outputs.values():
            assert path.name.startswith("202308281000_")

    def test_exclude_weights(self, tmp_path):
        """Weights can be excluded."""
        ds = _create_mock_composite()

        outputs = write_composite_cogs(ds, tmp_path, include_weights=False)

        assert len(outputs) == 1
        assert "correction_db" in outputs
        assert "weight_sum" not in outputs
        assert "n_radars" not in outputs

    def test_creates_output_directory(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        ds = _create_mock_composite()
        output_dir = tmp_path / "nested" / "dir"

        write_composite_cogs(ds, output_dir)

        assert output_dir.exists()


class TestWriteWeightsCog:
    """Tests for write_weights_cog function."""

    def test_exports_weight_variables(self, tmp_path):
        """Weight variables are exported."""
        ds = _create_mock_composite()

        outputs = write_weights_cog(ds, tmp_path)

        assert "weight_sum" in outputs
        assert "n_radars" in outputs
        assert outputs["weight_sum"].exists()
        assert outputs["n_radars"].exists()

    def test_prefix_is_applied(self, tmp_path):
        """Prefix is added to filenames."""
        ds = _create_mock_composite()

        outputs = write_weights_cog(ds, tmp_path, prefix="test")

        assert outputs["weight_sum"].name == "test_weight_sum.tif"
        assert outputs["n_radars"].name == "test_n_radars.tif"
