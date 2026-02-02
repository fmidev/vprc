"""Radar configuration loading from TOML.

This module provides functions to look up radar metadata from the
radar_defaults.toml configuration file.
"""

import os
import tomllib
from functools import lru_cache
from pathlib import Path


@lru_cache
def _load_radar_defaults() -> dict:
    """Load radar defaults from TOML config.

    Configuration precedence:
    1. TOML file specified by VPRC_RADAR_CONFIG environment variable
    2. Package-shipped radar_defaults.toml

    Note:
        Results are cached. Only loaded once per process.
    """
    # Check for user-specified config file
    config_path = os.environ.get('VPRC_RADAR_CONFIG')

    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(
                f"VPRC_RADAR_CONFIG points to non-existent file: {config_path}"
            )
    else:
        config_file = Path(__file__).parent / "radar_defaults.toml"

    with open(config_file, "rb") as f:
        return tomllib.load(f)


def get_radar_metadata(code: str) -> dict | None:
    """Get radar metadata by code (e.g., 'KAN', 'VIH').

    Args:
        code: Three-letter radar code (case-insensitive)

    Returns:
        Dict with keys like: name, antenna_height_m, latitude, longitude, etc.
        Returns None if radar code not found.

    Example:
        >>> meta = get_radar_metadata("KAN")
        >>> meta["name"]
        'KANKAANPAA'
        >>> meta["antenna_height_m"]
        174
    """
    config = _load_radar_defaults()
    return config.get(code.upper())


def get_radar_coords(code: str) -> tuple[float, float] | None:
    """Get (latitude, longitude) for a radar code.

    Args:
        code: Three-letter radar code (case-insensitive)

    Returns:
        Tuple of (latitude, longitude) in WGS84 degrees.
        Returns None if radar code not found or coordinates not available.

    Example:
        >>> get_radar_coords("KAN")
        (61.81085, 22.50204)
    """
    meta = get_radar_metadata(code)
    if meta and "latitude" in meta and "longitude" in meta:
        return (meta["latitude"], meta["longitude"])
    return None


def get_network_config() -> dict:
    """Get network-wide configuration settings.

    Returns:
        Dict with keys: crs_epsg, default_grid_resolution_m
    """
    config = _load_radar_defaults()
    return config.get("network", {})


def get_defaults() -> dict:
    """Get default values for radar parameters.

    These are used when a radar-specific value is not specified.

    Returns:
        Dict with keys: antenna_height_m, lowest_level_offset_m, beamwidth_deg, max_range_km
    """
    config = _load_radar_defaults()
    return config.get("defaults", {})


def list_radar_codes(enabled_only: bool = True) -> list[str]:
    """List all configured radar codes.

    Args:
        enabled_only: If True (default), only return radars with enabled=true.
            Set to False to include disabled radars.

    Returns:
        List of three-letter radar codes (e.g., ['VAN', 'KAN', 'VIH', ...])
    """
    config = _load_radar_defaults()
    defaults = config.get("defaults", {})
    default_enabled = defaults.get("enabled", True)

    # Exclude special sections
    special = {"network", "defaults"}
    codes = []
    for k, v in config.items():
        if k in special:
            continue
        if enabled_only:
            # Check if radar is enabled (defaults to default_enabled if not specified)
            if isinstance(v, dict) and v.get("enabled", default_enabled):
                codes.append(k)
        else:
            codes.append(k)
    return codes


@lru_cache(maxsize=1)
def _build_radar_name_to_code_map() -> dict[str, str]:
    """Build reverse mapping from full radar names to three-letter codes.

    This mapping is derived from the TOML configuration file, avoiding
    hardcoded name-to-code relationships in Python code.

    Returns:
        Dictionary mapping full radar names (uppercase) to codes.
        Example: {"KANKAANPAA": "KAN", "VANTAA": "VAN", ...}

    Note:
        Results are cached. Only built once per process.
        Reuses the cached TOML data from _load_radar_defaults().
    """
    radar_defaults = _load_radar_defaults()
    name_to_code = {}

    for code, config in radar_defaults.items():
        if code == 'other':
            continue
        # Get the 'name' field from config
        if isinstance(config, dict) and 'name' in config:
            name_to_code[config['name'].upper()] = code

    return name_to_code


def _radar_name_to_code(radar_name: str) -> str:
    """Convert full radar name to three-letter code.

    Args:
        radar_name: Full radar name from VVP file header (e.g., "KANKAANPAA")

    Returns:
        Three-letter radar code (e.g., "KAN")
        If not found in mapping, returns the radar_name unchanged
        (will fallback to 'other' in _get_radar_metadata).

    Example:
        >>> _radar_name_to_code("KANKAANPAA")
        'KAN'
        >>> _radar_name_to_code("UNKNOWN_RADAR")
        'UNKNOWN_RADAR'
    """
    name_map = _build_radar_name_to_code_map()
    return name_map.get(radar_name.upper(), radar_name)


def _get_radar_metadata(radar_code: str,
                        override_metadata: dict | None = None) -> dict:
    """Get radar metadata with proper precedence.

    Precedence (highest to lowest):
    1. Values provided in override_metadata parameter
    2. Values from radar-specific section in TOML
    3. Values from [defaults] section in TOML (used as fallback for unknown radars)

    Args:
        radar_code: Three-letter radar station code (e.g., 'KAN', 'VAN')
        override_metadata: Optional dict to override specific fields

    Returns:
        Dictionary with antenna_height_m, lowest_level_offset_m, beamwidth_deg,
        and any additional fields from override_metadata (e.g., freezing_level_m)

    Example:
        >>> meta = _get_radar_metadata('KAN', {'freezing_level_m': 2000})
        >>> meta['antenna_height_m']
        174
        >>> meta['freezing_level_m']
        2000
        >>> meta['beamwidth_deg']  # From [defaults] section
        0.95
    """
    # Load defaults from TOML
    radar_defaults = _load_radar_defaults()

    # Start with shared defaults (also serves as fallback for unknown radars)
    metadata = dict(radar_defaults.get('defaults', {}))

    # Get radar-specific config if it exists
    if radar_code in radar_defaults and radar_code != 'defaults':
        radar_config = radar_defaults[radar_code]
        # Update with radar-specific config (overrides defaults)
        metadata.update(radar_config)

    # Override with user-supplied values (highest priority)
    if override_metadata:
        metadata.update(override_metadata)

    return metadata
