"""Tests for COG export functionality."""

import numpy as np
import pytest
import xarray as xr

from vprc.io.geotiff import (
    prepare_for_export,
    write_composite_cogs,
    write_correction_cog,
    write_weights_cog,
)


def _create_mock_composite(crs_epsg: int = 3067) -> xr.Dataset:
    """Create a mock composite with distinct values in every cell.

    y ascends as in create_composite(), so placement errors show up.
    """
    x = np.arange(200_000, 250_000, 5_000)
    y = np.arange(6_700_000, 6_750_000, 5_000)

    correction = np.arange(len(y) * len(x), dtype=float).reshape(len(y), len(x))
    correction[0, 0] = np.nan  # A no-coverage cell

    weight_sum = correction[::-1] + 0.5
    n_radars = (np.arange(len(y) * len(x)) % 3).reshape(len(y), len(x)).astype(np.int8)

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
            "crs": f"EPSG:{crs_epsg}",
            "crs_epsg": crs_epsg,
            "max_range_km": 250.0,
            "radar_codes": ["KAN", "VIH"],
        },
    )

    return ds


class TestPrepareForExport:
    """Tests for prepare_for_export function."""

    def test_uses_dataset_crs(self):
        """The CRS comes from attrs["crs_epsg"], not the EPSG:3067 fallback."""
        ds_export = prepare_for_export(_create_mock_composite(crs_epsg=3035))
        assert ds_export.rio.crs.to_epsg() == 3035


class TestWriteCorrectionCog:
    """Tests for write_correction_cog function."""

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
        """GDAL marks files written with the COG layout."""
        import rasterio

        ds = _create_mock_composite()
        output_path = tmp_path / "test.tif"

        write_correction_cog(ds, output_path)

        with rasterio.open(output_path) as src:
            assert src.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"

    @pytest.mark.parametrize("compress", ["DEFLATE", "LZW", "ZSTD", "NONE"])
    def test_compression(self, tmp_path, compress):
        """Regression: NONE fell back to the driver's LZW default."""
        import rasterio

        ds = _create_mock_composite()
        output_path = tmp_path / "test.tif"

        write_correction_cog(ds, output_path, compress=compress)

        with rasterio.open(output_path) as src:
            written = src.compression.name.upper() if src.compression else "NONE"
            assert written == compress

    @pytest.mark.parametrize("variable", ["correction_db", "weight_sum", "n_radars"])
    def test_values_are_georeferenced(self, tmp_path, variable):
        """Each value is written at its own map coordinates."""
        import rasterio

        ds = _create_mock_composite(crs_epsg=3035)
        output_path = tmp_path / "test.tif"

        write_correction_cog(ds, output_path, variable=variable)

        with rasterio.open(output_path) as src:
            assert src.crs.to_epsg() == 3035
            # North-up: the first row is the northernmost.
            assert src.transform.e < 0
            assert src.transform.a > 0
            data = src.read(1)
            for j, i in [(0, 0), (0, 1), (3, 7), (9, 9)]:
                row, col = src.index(float(ds.x[i]), float(ds.y[j]))
                np.testing.assert_equal(data[row, col], ds[variable].values[j, i])

    def test_nodata_is_nan(self, tmp_path):
        """NaN values are preserved as nodata."""
        import rasterio

        ds = _create_mock_composite()
        output_path = tmp_path / "test.tif"

        write_correction_cog(ds, output_path)

        with rasterio.open(output_path) as src:
            src.read(1)
            # Float output marks missing values with NaN nodata
            assert src.nodata is not None
            assert np.isnan(src.nodata)

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

        outputs = write_composite_cogs(ds, tmp_path, include_weights=True)

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

    def test_prefix_is_applied(self, tmp_path):
        """Prefix is added to filenames."""
        ds = _create_mock_composite()

        outputs = write_weights_cog(ds, tmp_path, prefix="test")

        assert outputs["weight_sum"].name == "test_weight_sum.tif"
        assert outputs["n_radars"].name == "test_n_radars.tif"


class TestEmptyCompositeExport:
    """Tests for exporting empty composites (no precipitation scenario)."""

    def _create_empty_composite(self) -> xr.Dataset:
        """Create an empty composite with all NaN corrections."""
        x = np.arange(200_000, 300_000, 10_000)
        y = np.arange(6_700_000, 6_800_000, 10_000)

        ds = xr.Dataset(
            {
                "correction_db": (["y", "x"], np.full((len(y), len(x)), np.nan)),
                "weight_sum": (["y", "x"], np.zeros((len(y), len(x)))),
                "n_radars": (["y", "x"], np.zeros((len(y), len(x)), dtype=np.int8)),
            },
            coords={
                "x": x,
                "y": y,
            },
            attrs={
                "crs": "EPSG:3067",
                "crs_epsg": 3067,
                "max_range_km": 0.0,
                "radar_codes": ["KAN", "VIH"],
                "empty_composite": True,
            },
        )
        return ds

    def test_empty_composite_creates_valid_cog(self, tmp_path):
        """Empty composite can be exported to valid COG."""
        import rasterio

        ds = self._create_empty_composite()
        output_path = tmp_path / "empty_correction.tif"

        write_correction_cog(ds, output_path)

        assert output_path.exists()

        with rasterio.open(output_path) as src:
            assert src.driver == "GTiff"
            assert src.crs.to_epsg() == 3067
            assert src.count == 1
            data = src.read(1)
            # All values should be NaN (nodata)
            assert np.all(np.isnan(data))

    def test_empty_composite_weight_sum_is_zero(self, tmp_path):
        """Weight sum in empty composite is all zeros."""
        import rasterio

        ds = self._create_empty_composite()
        output_path = tmp_path / "weight_sum.tif"

        write_correction_cog(ds, output_path, variable="weight_sum")

        with rasterio.open(output_path) as src:
            data = src.read(1)
            assert np.all(data == 0)

    def test_empty_composite_n_radars_is_zero(self, tmp_path):
        """n_radars in empty composite is all zeros."""
        import rasterio

        ds = self._create_empty_composite()
        output_path = tmp_path / "n_radars.tif"

        write_correction_cog(ds, output_path, variable="n_radars")

        with rasterio.open(output_path) as src:
            data = src.read(1)
            assert np.all(data == 0)
