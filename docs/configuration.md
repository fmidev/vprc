# Configuration Guide

## Overview

The `vprc` package uses TOML-based configuration for radar station metadata. All radar name-to-code mappings come from the configuration file, with no hardcoded values in Python code.

## TOML File Structure

```toml
[network]  # Compositing settings
crs_epsg = 3067
default_grid_resolution_m = 1000

[defaults]  # Shared defaults for values a radar section omits
beamwidth_deg = 0.95
horizon_angle_deg = 0.0
max_range_km = 250
enabled = true

[KAN]  # Three-letter code (section key)
name = "KANKAANPAA"  # Full radar name from VVP file headers
antenna_height_m = 174
lowest_level_offset_m = 126
latitude = 61.81085
longitude = 22.50204

[VAN]
name = "VANTAA"
antenna_height_m = 83
lowest_level_offset_m = 17

[KOR]
name = "KORPPOO"
aliases = ["KOR"]  # Older VVP header name
antenna_height_m = 61
lowest_level_offset_m = 39
```

### Fields

**Network (`[network]`):**
- `crs_epsg` (int): Composite grid coordinate reference system (3067,
  ETRS-TM35FIN)
- `default_grid_resolution_m` (int): Composite grid resolution in meters

**Shared defaults:**
- `beamwidth_deg` (float): One-way half-power beamwidth in degrees
  - Defined in `[defaults]` section
  - Can be overridden per-radar
- `horizon_angle_deg` (float): Lowest unblocked beam angle in degrees
  - Defined in `[defaults]` section
  - Can be overridden per-radar for terrain-blocked or below-horizontal coverage
- `max_range_km` (int): Maximum usable range for compositing
- `enabled` (bool): Whether the radar is active in the network;
  `list_radar_codes()` returns only enabled radars unless called with
  `enabled_only=False`

**Required (per-radar):**
- `name` (string): Full radar name for VVP header matching (case-insensitive)
- `antenna_height_m` (int): Antenna elevation above sea level (meters)
- `lowest_level_offset_m` (int): Height of the lowest profile level above the
  antenna (meters), the legacy `$alintaso`. VVP levels are at 100, 300, 500, …
  m ASL, so this is normally `(100 - antenna_height_m) mod 200`.

The two geometry values have no default: a radar section missing either
raises `ValueError` when its metadata is resolved, and geometry in
`[defaults]` is ignored with a warning. Profiles always start at
`lowest_level_offset_m`; a lowest level missing from the file becomes a no-echo
level. Files whose levels are not on the 200 m grid through that level raise
`ValueError`, which usually means the geometry is wrong.

**Optional (per-radar):**
- `aliases` (list of strings): Other VVP header names for the same radar, such
  as an older IRIS site name. A name may belong to only one radar.
- `beamwidth_deg` (float): Override default beamwidth for this specific radar
- `horizon_angle_deg` (float): Override the default terrain horizon
- `latitude`, `longitude` (float): WGS84 radar location, required for
  compositing. `get_radar_coords()` returns `None` for a radar without them
  (currently KER).
- `max_range_km`, `enabled`: Override the defaults above

**Optional (runtime):**
- `freezing_level_m` (float | None): Freezing level from NWP data (meters ASL)
  - `None`: Unknown (not retrieved)
  - At or below the antenna, including `0` or negative: no melting layer
    above the antenna → stored as 0
  - Above the antenna: stored in the dataset attributes in meters above the
    antenna, like the `height` coordinate
  - Non-finite values raise `ValueError`

## Configuration Precedence

`VPRC_RADAR_CONFIG` selects the TOML file if set; otherwise the package default
`src/vprc/radar_defaults.toml` is used. Within that file, values are resolved
in this order (highest to lowest):

1. **Function parameters** - `read_vvp(radar_metadata={...})`
2. **Radar-specific TOML section** - e.g., `[KAN]`
3. **Shared defaults** - `[defaults]` section in TOML, for values the radar section omits
   (never `antenna_height_m` or `lowest_level_offset_m`)

## Usage Examples

### Accessing Radar Metadata

```python
from vprc import get_radar_metadata, get_radar_coords, list_radar_codes

# Get all metadata for a radar
meta = get_radar_metadata("KAN")
print(meta)  # {'name': 'KANKAANPAA', 'antenna_height_m': 174, ...}

# Get just coordinates
lat, lon = get_radar_coords("VIH")
print(f"Vihti radar: {lat}°N, {lon}°E")

# List all configured radars
for code in list_radar_codes():
    coords = get_radar_coords(code)
    print(f"{code}: {coords}")
```

### Basic VVP Processing

```python
from vprc import read_vvp

# Uses package defaults from radar_defaults.toml
ds = read_vvp("202508241100_KAN.VVP_40.txt")
print(ds.attrs["antenna_height_m"])  # 174
print(ds.attrs["beamwidth_deg"])  # 0.95 (from [defaults])
```

### Override Specific Fields

```python
# Override freezing level from NWP data
ds = read_vvp("202508241100_KAN.VVP_40.txt", {"freezing_level_m": 2000})

# Override beamwidth for a specific radar
ds = read_vvp("202508241100_KAN.VVP_40.txt", {"beamwidth_deg": 1.0})

# Geometry overrides must come as a pair; overriding only one raises ValueError
ds = read_vvp(
    "202508241100_KAN.VVP_40.txt",
    {
        "antenna_height_m": 174,
        "lowest_level_offset_m": 126,
    },
)
```

### Custom Configuration File

```python
import os

os.environ["VPRC_RADAR_CONFIG"] = "/path/to/custom_radars.toml"
ds = read_vvp("202508241100_KAN.VVP_40.txt")
```

Example custom TOML with radar-specific beamwidth:

```toml
[defaults]
beamwidth_deg = 0.95

[KAN]
name = "KANKAANPAA"
antenna_height_m = 174
lowest_level_offset_m = 126
beamwidth_deg = 1.0  # Override for this specific radar

[VAN]
name = "VANTAA"
antenna_height_m = 83
lowest_level_offset_m = 17
# Uses default beamwidth_deg = 0.95
```

## Environment Variables

**`VPRC_RADAR_CONFIG`**: Path to custom TOML configuration file
- If not set, uses package default `radar_defaults.toml`
- Raises `FileNotFoundError` if path doesn't exist
- The configuration is loaded once per process and cached, so set the variable
  before the first `vprc` call that reads configuration; changing it later in
  the same process has no effect

```bash
export VPRC_RADAR_CONFIG=/path/to/custom_radars.toml
python your_script.py
```

## Data Flow

```
VVP file "KANKAANPAA" → TOML name lookup → "KAN" → load metadata
                              ↓
                    radar_defaults.toml
                    (or VPRC_RADAR_CONFIG)
```

A VVP header name that matches no radar `name` or `aliases` entry raises
`ValueError`. `[defaults]` is not used for unknown radars: its antenna height
would shift every profile level and produce wrong corrections.

## Radar Stations (Package Default)

Antenna heights and level offsets come from the legacy `allprof_prodx2.pl`
radar table. Korppoo VVP headers say `KORPPOO`; the legacy name `KOR` is kept
as an alias. Coordinates are in `radar_defaults.toml`.

| Code | Name         | Height (m) | Offset (m) |
|------|--------------|------------|------------|
| VAN  | VANTAA       | 83         | 17         |
| KER  | KERAVA       | 83         | 5          |
| IKA  | IKAALINEN    | 154        | 146        |
| KAN  | KANKAANPAA   | 174        | 126        |
| KES  | KESALAHTI    | 174        | 126        |
| PET  | PETAJAVESI   | 271        | 29         |
| ANJ  | ANJALANKOSKI | 139        | 161        |
| KUO  | KUOPIO       | 268        | 32         |
| KOR  | KORPPOO      | 61         | 39         |
| UTA  | UTAJARVI     | 118        | 182        |
| LUO  | LUOSTO       | 530        | 170        |
| VIM  | VIMPELI      | 198        | 102        |
| NUR  | NURMES       | 323        | 177        |
| VIH  | VIHTI        | 181        | 119        |
| KAU  | KAUNISPAA    | 489        | 11         |
