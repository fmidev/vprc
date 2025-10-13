# Configuration Guide

## Overview

The `vprc` package uses TOML-based configuration for radar station metadata. All radar name-to-code mappings come from the configuration file, with no hardcoded values in Python code.

## TOML File Structure

```toml
[KAN]  # Three-letter code (section key)
name = "KANKAANPAA"  # Full radar name from VVP file headers
antenna_height_m = 174
lowest_level_offset_m = 126

[VAN]
name = "VANTAA"
antenna_height_m = 83
lowest_level_offset_m = 17

[other]  # Fallback for unknown radars
antenna_height_m = 198
lowest_level_offset_m = 102
```

### Fields

**Required (in TOML):**
- `name` (string): Full radar name for VVP header matching (case-insensitive)
- `antenna_height_m` (int): Antenna elevation above sea level (meters)
- `lowest_level_offset_m` (int): Offset from nearest profile level (meters)

**Optional (runtime):**
- `freezing_level_m` (int | None): Freezing level from NWP data
  - `None`: Unknown (not retrieved)
  - `0` or negative: No freezing layer → normalized to 0
  - Positive: Valid freezing level in meters above antenna

## Configuration Precedence

Values are resolved in this order (highest to lowest):

1. **Function parameters** - `read_vvp(radar_metadata={...})`
2. **Environment variable** - `VPRC_RADAR_CONFIG` points to custom TOML
3. **Package default** - `src/vprc/radar_defaults.toml`
4. **Fallback** - `[other]` section if radar code not found

## Usage Examples

### Basic Usage

```python
from vprc import read_vvp

# Uses package defaults
ds = read_vvp('202508241100_KAN.VVP_40.txt')
print(ds.attrs['antenna_height_m'])  # 174
```

### Override Specific Fields

```python
# Override freezing level from NWP data
ds = read_vvp('202508241100_KAN.VVP_40.txt', {
    'freezing_level_m': 2000
})
```

### Custom Configuration File

```python
import os

os.environ['VPRC_RADAR_CONFIG'] = '/path/to/custom_radars.toml'
ds = read_vvp('202508241100_KAN.VVP_40.txt')
```

### Full Precedence Example

```python
import os

# Custom TOML: [KAN] antenna_height_m = 500
os.environ['VPRC_RADAR_CONFIG'] = '/opt/fmi/radars.toml'

# Function param wins
ds = read_vvp('202508241100_KAN.VVP_40.txt', {
    'antenna_height_m': 300,      # Overrides TOML
    'freezing_level_m': 2000      # New field
})
# Result: antenna_height_m=300, freezing_level_m=2000, lowest_level_offset_m from TOML
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

## API Functions

### Public API

```python
from vprc import read_vvp

ds = read_vvp(filepath, radar_metadata=None)
```

### Internal Functions (for testing)

```python
from vprc.io import (
    _load_radar_defaults,           # Load TOML config (cached)
    _build_radar_name_to_code_map,  # Build name→code mapping (cached)
    _radar_name_to_code,            # Convert "KANKAANPAA" → "KAN"
    _get_radar_metadata,            # Get metadata with overrides
)

# Clear cache (needed when changing VPRC_RADAR_CONFIG)
_load_radar_defaults.cache_clear()
_build_radar_name_to_code_map.cache_clear()
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

## Adding New Radars

1. Add section to TOML:
   ```toml
   [NEW]
   name = "NEWRADAR"
   antenna_height_m = 250
   lowest_level_offset_m = 50
   ```

2. Clear cache (if needed):
   ```python
   _load_radar_defaults.cache_clear()
   _build_radar_name_to_code_map.cache_clear()
   ```

3. Use immediately - no Python code changes required.

## Common Patterns

### Airflow Deployment

```python
from airflow.decorators import task

@task.docker(image="fmi/vprc:latest")
def process_vpr(vvp_file: str, nwp_freezing_level: int):
    from vprc import read_vvp

    ds = read_vvp(vvp_file, {
        'freezing_level_m': nwp_freezing_level
    })
    # ... processing ...
```

### Testing

```python
import os
from vprc.io import _load_radar_defaults, _build_radar_name_to_code_map

os.environ['VPRC_RADAR_CONFIG'] = 'tests/data/test_radars.toml'
_load_radar_defaults.cache_clear()
_build_radar_name_to_code_map.cache_clear()

ds = read_vvp('tests/data/test_file.txt')
```
