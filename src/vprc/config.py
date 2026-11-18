"""Radar configuration loading from TOML.

This module provides functions to look up radar metadata from the
radar_defaults.toml configuration file.
"""

import logging
import os
import tomllib
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Profile geometry must come from one source: a radar section or an override.
# Mixing an antenna height with another source's offset misplaces every level.
GEOMETRY_KEYS = ("antenna_height_m", "lowest_level_offset_m")


def _radar_config_path() -> Path:
    """Return the selected radar configuration file path."""
    config_path = os.environ.get("VPRC_RADAR_CONFIG")
    if config_path:
        return Path(config_path)
    return Path(__file__).parent / "radar_defaults.toml"


@lru_cache
def _load_radar_defaults() -> dict:
    """Load radar defaults from TOML config.

    Configuration precedence:
    1. TOML file specified by VPRC_RADAR_CONFIG environment variable
    2. Package-shipped radar_defaults.toml

    Note:
        Results are cached. Only loaded once per process.
    """
    config_file = _radar_config_path()
    if os.environ.get("VPRC_RADAR_CONFIG") and not config_file.exists():
        raise FileNotFoundError(
            f"VPRC_RADAR_CONFIG points to non-existent file: {config_file}"
        )

    with open(config_file, "rb") as f:
        config = tomllib.load(f)
    ignored = [key for key in GEOMETRY_KEYS if key in config.get("defaults", {})]
    if ignored:
        logger.warning(
            "Ignoring %s in [defaults] of %s; radar geometry must be defined "
            "in each radar section",
            ", ".join(ignored),
            config_file,
        )
    return config


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

    These are used when a radar-specific value is not specified. Radar
    geometry (antenna height and lowest level offset) has no default.

    Returns:
        Dict with keys such as beamwidth_deg, horizon_angle_deg, max_range_km
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
    """Build reverse mapping from VVP header radar names to three-letter codes.

    Each radar section's ``name`` and optional ``aliases`` are mapped, so a
    station whose IRIS site name has changed (e.g. Korppoo: ``KOR`` in older
    files, ``KORPPOO`` later) resolves to the same configuration.

    Returns:
        Dictionary mapping header names (uppercase) to codes.
        Example: {"KANKAANPAA": "KAN", "VANTAA": "VAN", ...}

    Raises:
        ValueError: If two radar sections claim the same header name.

    Note:
        Results are cached. Only built once per process.
        Reuses the cached TOML data from _load_radar_defaults().
    """
    radar_defaults = _load_radar_defaults()
    name_to_code: dict[str, str] = {}

    for code, config in radar_defaults.items():
        if code == "other" or not isinstance(config, dict) or "name" not in config:
            continue
        aliases = config.get("aliases", [])
        if isinstance(aliases, str) or not isinstance(aliases, list):
            raise TypeError(f"Radar {code}: 'aliases' must be a list of names")
        for name in (config["name"], *aliases):
            key = name.upper()
            if name_to_code.get(key, code) != code:
                raise ValueError(
                    f"Radar header name {key!r} is configured for both "
                    f"{name_to_code[key]} and {code}"
                )
            name_to_code[key] = code

    return name_to_code


def _radar_name_to_code(radar_name: str) -> str:
    """Convert full radar name to three-letter code.

    Args:
        radar_name: Full radar name from VVP file header (e.g., "KANKAANPAA")

    Returns:
        Three-letter radar code (e.g., "KAN")
        If not found in mapping, returns the radar_name unchanged.
        Use resolve_radar_code() where an unknown name must be an error.

    Example:
        >>> _radar_name_to_code("KANKAANPAA")
        'KAN'
        >>> _radar_name_to_code("UNKNOWN_RADAR")
        'UNKNOWN_RADAR'
    """
    name_map = _build_radar_name_to_code_map()
    return name_map.get(radar_name.upper(), radar_name)


def resolve_radar_code(radar_name: str) -> str:
    """Return the configured radar code for a VVP header radar name.

    Unlike _radar_name_to_code(), an unknown name is an error: falling back to
    [defaults] would apply the wrong antenna height and shift every profile
    level, producing plausible-looking but wrong corrections.

    Raises:
        ValueError: If no radar section has this name or alias.
    """
    code = _build_radar_name_to_code_map().get(radar_name.upper())
    if code is None:
        raise ValueError(
            f"Radar name {radar_name!r} from the VVP header is not configured "
            f"as a radar name or alias in {_radar_config_path()}"
        )
    return code


def _get_radar_metadata(radar_code: str,
                        override_metadata: dict | None = None) -> dict:
    """Get radar metadata with proper precedence.

    Precedence (highest to lowest):
    1. Values provided in override_metadata parameter
    2. Values from radar-specific section in TOML
    3. Values from [defaults] section in TOML, except radar geometry

    ``antenna_height_m`` and ``lowest_level_offset_m`` must both come from
    the radar section or both from override_metadata.

    Args:
        radar_code: Three-letter radar station code (e.g., 'KAN', 'VAN')
        override_metadata: Optional dict to override specific fields

    Returns:
        Dictionary with antenna_height_m, lowest_level_offset_m, beamwidth_deg,
        and any additional fields from override_metadata (e.g., freezing_level_m)

    Raises:
        ValueError: If only one geometry value is overridden, or the radar
            section does not define both and they are not overridden.

    Example:
        >>> meta = _get_radar_metadata('KAN', {'freezing_level_m': 2000})
        >>> meta['antenna_height_m']
        174
        >>> meta['freezing_level_m']
        2000
        >>> meta['beamwidth_deg']  # From [defaults] section
        0.95
    """
    radar_defaults = _load_radar_defaults()
    overrides = override_metadata or {}
    overridden = [key for key in GEOMETRY_KEYS if key in overrides]
    if len(overridden) == 1:
        raise ValueError(
            "antenna_height_m and lowest_level_offset_m must be overridden "
            f"together; only {overridden[0]} was given"
        )

    # Shared defaults never provide geometry.
    metadata = {
        key: value
        for key, value in radar_defaults.get("defaults", {}).items()
        if key not in GEOMETRY_KEYS
    }

    # Get radar-specific config if it exists
    if radar_code in radar_defaults and radar_code not in ("defaults", "network"):
        radar_config = radar_defaults[radar_code]
        # Update with radar-specific config (overrides defaults). Header
        # aliases are lookup configuration, not radar metadata.
        metadata.update(
            {key: value for key, value in radar_config.items() if key != "aliases"}
        )

    if not overridden:
        missing = [key for key in GEOMETRY_KEYS if key not in metadata]
        if missing:
            raise ValueError(
                f"Radar {radar_code} does not define {' and '.join(missing)} "
                f"in {_radar_config_path()}"
            )

    # Override with user-supplied values (highest priority)
    metadata.update(overrides)

    return metadata
