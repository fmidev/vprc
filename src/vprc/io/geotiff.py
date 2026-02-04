"""Export functions for VPR correction products.

This module provides functions to export composite VPR corrections
as Cloud Optimized GeoTIFFs (COG).
"""

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

# rioxarray import activates the .rio accessor on xarray objects
import rioxarray  # noqa: F401


# Compression options supported by GDAL COG driver
Compression = Literal["DEFLATE", "LZW", "ZSTD", "NONE"]

# Default compression settings
DEFAULT_COMPRESSION: Compression = "DEFLATE"
DEFAULT_BLOCKSIZE = 512


def prepare_for_export(ds: xr.Dataset) -> xr.Dataset:
    """Prepare composite Dataset for geospatial export.

    Adds CF-compliant CRS metadata and spatial dimension info
    required by rioxarray for proper GeoTIFF export.

    Args:
        ds: Composite Dataset from composite_corrections()

    Returns:
        Dataset with CRS and spatial dimensions set
    """
    # Write CRS from existing attribute
    crs_epsg = ds.attrs.get("crs_epsg", 3067)
    ds = ds.rio.write_crs(crs_epsg)
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")

    return ds


def write_correction_cog(
    ds: xr.Dataset,
    path: str | Path,
    variable: str = "correction_db",
    compress: Compression = DEFAULT_COMPRESSION,
    blocksize: int = DEFAULT_BLOCKSIZE,
) -> None:
    """Export a single correction variable as Cloud Optimized GeoTIFF.

    Args:
        ds: Composite Dataset from composite_corrections()
        path: Output file path (.tif)
        variable: Variable name to export
        compress: Compression method (DEFLATE, LZW, ZSTD, NONE)
        blocksize: Tile size for COG (default 512)

    Raises:
        KeyError: If variable not found in dataset
    """
    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found in dataset. "
                      f"Available: {list(ds.data_vars)}")

    # Ensure CRS metadata is set
    ds = prepare_for_export(ds)

    # Get DataArray and set nodata for NaN values
    da = ds[variable]

    # Only set NaN nodata for floating-point types
    # Integer types (like n_radars) don't support NaN
    if np.issubdtype(da.dtype, np.floating):
        da = da.rio.write_nodata(np.nan)

    # Build COG driver options
    driver_options = {
        "driver": "COG",
        "blocksize": blocksize,
    }

    if compress != "NONE":
        driver_options["compress"] = compress
        # Use floating-point predictor for better compression of dB values
        if compress in ("DEFLATE", "LZW", "ZSTD"):
            driver_options["predictor"] = "YES"

    da.rio.to_raster(path, **driver_options)


def write_weights_cog(
    ds: xr.Dataset,
    output_dir: str | Path,
    prefix: str = "",
    compress: Compression = DEFAULT_COMPRESSION,
) -> dict[str, Path]:
    """Export weight and metadata variables as separate COGs.

    Exports weight_sum and n_radars as separate GeoTIFFs.

    Args:
        ds: Composite Dataset from composite_corrections()
        output_dir: Directory for output files
        prefix: Optional filename prefix (e.g., timestamp)
        compress: Compression method

    Returns:
        Dict mapping variable names to output paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_vars = ["weight_sum", "n_radars"]
    outputs = {}

    for var in weight_vars:
        if var not in ds:
            continue

        filename = f"{prefix}_{var}.tif" if prefix else f"{var}.tif"
        path = output_dir / filename

        write_correction_cog(ds, path, variable=var, compress=compress)
        outputs[var] = path

    return outputs


def write_composite_cogs(
    ds: xr.Dataset,
    output_dir: str | Path,
    prefix: str = "",
    compress: Compression = DEFAULT_COMPRESSION,
    include_weights: bool = False,
) -> dict[str, Path]:
    """Export composite products as separate COG files.

    Creates separate GeoTIFFs for:
    - correction_db (main composite correction)
    - weight_sum (optional; total weight at each point)
    - n_radars (optional; number of contributing radars)

    Args:
        ds: Composite Dataset from composite_corrections()
        output_dir: Directory for output files
        prefix: Optional filename prefix (e.g., "202308281000")
        compress: Compression method (DEFLATE, LZW, ZSTD, NONE)
        include_weights: Whether to export weight variables

    Returns:
        Dict mapping variable names to output file paths

    Example:
        >>> ds = composite_corrections(radars, grid)
        >>> paths = write_composite_cogs(
        ...     ds, "/output", prefix="202308281000", include_weights=True
        ... )
        >>> print(paths)
        {'correction_db': Path('/output/202308281000_correction_db.tif'),
         'weight_sum': Path('/output/202308281000_weight_sum.tif'),
         'n_radars': Path('/output/202308281000_n_radars.tif')}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # Main correction variable
    correction_var = "correction_db"
    if correction_var in ds:
        filename = f"{prefix}_{correction_var}.tif" if prefix else f"{correction_var}.tif"
        path = output_dir / filename
        write_correction_cog(ds, path, variable=correction_var, compress=compress)
        outputs[correction_var] = path

    # Weight variables
    if include_weights:
        weight_outputs = write_weights_cog(ds, output_dir, prefix, compress)
        outputs.update(weight_outputs)

    return outputs
