"""Radar configuration loading from TOML.

This module provides functions to look up radar metadata from the
radar_defaults.toml configuration file.
"""

import tomllib
from functools import lru_cache
from pathlib import Path


@lru_cache
def _load_radar_defaults() -> dict:
    """Load radar defaults from TOML config."""
    config_path = Path(__file__).parent / "radar_defaults.toml"
    with open(config_path, "rb") as f:
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


def list_radar_codes() -> list[str]:
    """List all configured radar codes.

    Returns:
        List of three-letter radar codes (e.g., ['VAN', 'KAN', 'VIH', ...])
    """
    config = _load_radar_defaults()
    # Exclude special sections
    special = {"network", "defaults"}
    return [k for k in config.keys() if k not in special]
