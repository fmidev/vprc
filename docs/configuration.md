# Configuration Guide

## Overview

The `vprc` package uses TOML-based configuration for radar station metadata. All radar name-to-code mappings come from the configuration file, with no hardcoded values in Python code.

## TOML File Structure

```toml
[defaults]  # Shared defaults and fallback for unknown radars
antenna_height_m = 198
lowest_level_offset_m = 102
beamwidth_deg = 0.95

[KAN]  # Three-letter code (section key)
name = "KANKAANPAA"  # Full radar name from VVP file headers
antenna_height_m = 174
lowest_level_offset_m = 126

[VAN]
name = "VANTAA"
antenna_height_m = 83
lowest_level_offset_m = 17
```

### Fields

**Shared defaults:**
- `beamwidth_deg` (float): One-way half-power beamwidth in degrees
  - Defined in `[defaults]` section
  - Can be overridden per-radar

**Required (per-radar):**
- `name` (string): Full radar name for VVP header matching (case-insensitive)
- `antenna_height_m` (int): Antenna elevation above sea level (meters)
- `lowest_level_offset_m` (int): Offset from nearest profile level (meters)

**Optional (per-radar):**
- `beamwidth_deg` (float): Override default beamwidth for this specific radar

**Optional (runtime):**
- `freezing_level_m` (int | None): Freezing level from NWP data (meters ASL)
  - `None`: Unknown (not retrieved)
  - `0` or negative: No freezing layer → normalized to 0
  - Positive: Valid freezing level in meters above sea level

## Configuration Precedence

Values are resolved in this order (highest to lowest):

1. **Function parameters** - `read_vvp(radar_metadata={...})`
2. **Radar-specific TOML section** - e.g., `[KAN]`
3. **Shared defaults** - `[defaults]` section in TOML (also serves as fallback for unknown radars)
4. **Environment variable** - `VPRC_RADAR_CONFIG` points to custom TOML
5. **Package default** - `src/vprc/radar_defaults.toml`

## Usage Examples

### Accessing Radar Metadata

```python
from vprc import get_radar_metadata, get_radar_coords, list_radar_codes

# Get all metadata for a radar
meta = get_radar_metadata('KAN')
print(meta)  # {'name': 'KANKAANPAA', 'antenna_height_m': 174, ...}

# Get just coordinates
lat, lon = get_radar_coords('VIH')
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
ds = read_vvp('202508241100_KAN.VVP_40.txt')
print(ds.attrs['antenna_height_m'])  # 174
print(ds.attrs['beamwidth_deg'])     # 0.95 (from [defaults])
```

### Override Specific Fields

```python
# Override freezing level from NWP data
ds = read_vvp('202508241100_KAN.VVP_40.txt', {
    'freezing_level_m': 2000
})

# Override beamwidth for a specific radar
ds = read_vvp('202508241100_KAN.VVP_40.txt', {
    'beamwidth_deg': 1.0
})
```

### Custom Configuration File

```python
import os

os.environ['VPRC_RADAR_CONFIG'] = '/path/to/custom_radars.toml'
ds = read_vvp('202508241100_KAN.VVP_40.txt')
```

Example custom TOML with radar-specific beamwidth:

```toml
[defaults]
beamwidth_deg = 0.95
# Shared defaults and fallback for unknown radars
antenna_height_m = 198
lowest_level_offset_m = 102

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

## Radar Stations (Package Default)

Based on `allprof_prodx2.pl` lines 32-50, 75-135:

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
| KOR  | KOR          | 61         | 39         |
| UTA  | UTAJARVI     | 118        | 182        |
| LUO  | LUOSTO       | 530        | 170        |
| VIM  | VIMPELI      | 198        | 102        |
| NUR  | NURMES       | 323        | 177        |
| VIH  | VIHTI        | 181        | 119        |
| KAU  | KAUNISPAA    | 489        | 11         |
