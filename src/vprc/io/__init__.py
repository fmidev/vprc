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

# Re-export config functions for backward compatibility
from ..config import (
    _build_radar_name_to_code_map,
    _get_radar_metadata,
    _load_radar_defaults,
    _radar_name_to_code,
)
from ..exceptions import NoUsableProfileError
from .geotiff import (
    Compression,
    prepare_for_export,
    write_composite_cogs,
    write_correction_cog,
    write_weights_cog,
)
from .vvp import (
    VVPHeader,
    _parse_vvp_file,
    _parse_vvp_header,
    _vvp_dataframe_to_xarray,
    height_to_sea_level,
    read_vvp,
)

__all__ = [
    # VVP parsing
    "VVPHeader",
    "NoUsableProfileError",
    "read_vvp",
    "height_to_sea_level",
    # GeoTIFF export
    "Compression",
    "prepare_for_export",
    "write_correction_cog",
    "write_composite_cogs",
    "write_weights_cog",
]
