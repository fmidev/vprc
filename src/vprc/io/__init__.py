"""I/O functions for VPR correction.

Public API for reading VVP profiles and exporting results.

VVP parsing:
    read_vvp() - Parse VVP profile files to xarray Dataset
    height_to_sea_level() - Convert heights above antenna to sea level

GeoTIFF export:
    prepare_for_export() - Add CRS metadata for geospatial export
    write_correction_cog() - Export single variable as Cloud Optimized GeoTIFF
    write_composite_cogs() - Export all composite products as COGs
    write_weights_cog() - Export weight variables as COGs
"""

from .vvp import (
    VVPHeader,
    read_vvp,
    height_to_sea_level,
    # Internal functions exposed for testing
    _load_radar_defaults,
    _build_radar_name_to_code_map,
    _radar_name_to_code,
    _get_radar_metadata,
    _parse_vvp_header,
    _parse_vvp_file,
    _vvp_dataframe_to_xarray,
)

from .geotiff import (
    Compression,
    prepare_for_export,
    write_correction_cog,
    write_composite_cogs,
    write_weights_cog,
)

__all__ = [
    # VVP parsing
    "VVPHeader",
    "read_vvp",
    "height_to_sea_level",
    # GeoTIFF export
    "Compression",
    "prepare_for_export",
    "write_correction_cog",
    "write_composite_cogs",
    "write_weights_cog",
]
