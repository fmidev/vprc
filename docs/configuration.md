# Configuration Guide

## Radar Station Configuration

The `vprc` package uses radar-specific metadata (antenna height, profile offsets, etc.) for VPR correction calculations. Configuration follows a clear precedence order to balance operational flexibility with sensible defaults.

## Configuration Precedence

Configuration is resolved in this order (highest to lowest priority):

1. **Function parameters** - Values passed directly to `read_vvp()`
2. **Environment variable TOML** - Custom config via `VPRC_RADAR_CONFIG`
3. **Package defaults** - Built-in `radar_defaults.toml`
4. **Fallback** - `[other]` section if radar code not found

## Usage Examples

### Basic Usage (Package Defaults)

The simplest approach - uses built-in radar configurations:

```python
from vprc import read_vvp

# Automatically loads KAN radar defaults from radar_defaults.toml
ds = read_vvp('202508241100_KAN.VVP_40.txt')

print(ds.attrs['antenna_height_m'])        # 174 (from TOML)
print(ds.attrs['lowest_level_offset_m'])   # 126 (from TOML)
```

### Override Specific Fields

Common in operational workflows where you need to inject runtime data (e.g., freezing level from NWP):

```python
from vprc import read_vvp

# Override freezing level from MEPS data, keep other defaults
ds = read_vvp('202508241100_KAN.VVP_40.txt', {
    'freezing_level_m': 2000  # From NWP model
})

print(ds.attrs['antenna_height_m'])        # 174 (from TOML)
print(ds.attrs['freezing_level_m'])        # 2000 (from parameter)
```

### Custom Configuration File

For testing or alternative radar networks:

```python
import os
from vprc import read_vvp

# Point to custom radar configuration
os.environ['VPRC_RADAR_CONFIG'] = '/path/to/test_radars.toml'

ds = read_vvp('202508241100_KAN.VVP_40.txt')
# Now loads from custom TOML instead of package default
```

### Full Override Example

Combining all precedence levels:

```python
import os
from vprc import read_vvp

# Custom TOML provides base config
os.environ['VPRC_RADAR_CONFIG'] = '/opt/fmi/radars.toml'

# Function parameter overrides specific field
ds = read_vvp('202508241100_KAN.VVP_40.txt', {
    'antenna_height_m': 180,      # Overrides TOML value
    'freezing_level_m': 2100      # New field not in TOML
})

# Result:
# - antenna_height_m: 180 (function param)
# - freezing_level_m: 2100 (function param)
# - lowest_level_offset_m: from VPRC_RADAR_CONFIG file
# - other fields: from VPRC_RADAR_CONFIG file
```

## Configuration Fields

### Required Fields (in TOML)

- `antenna_height_m` (int): Antenna elevation above sea level (meters)
- `lowest_level_offset_m` (int): Offset from nearest profile level (meters)

### Optional Runtime Fields

- `freezing_level_m` (int | None):
  - `None`: Unknown (not yet retrieved from NWP)
  - `0` or negative: No freezing layer → normalized to 0
  - Positive: Valid freezing level in meters above antenna

## TOML File Format

Example `radar_defaults.toml` structure:

```toml
[KAN]  # Kankaanpää radar
antenna_height_m = 174
lowest_level_offset_m = 126

[VAN]  # Vantaa radar
antenna_height_m = 83
lowest_level_offset_m = 17

[other]  # Fallback for unknown radars
antenna_height_m = 198
lowest_level_offset_m = 102
```

## Airflow Integration Pattern

Typical operational deployment:

```python
from airflow.decorators import task

@task.docker(image="fmi/vprc:latest")
def correct_vpr(vvp_file: str, nwp_data: dict) -> str:
    """
    VPR correction task.

    - Base radar config from package TOML
    - Freezing level from NWP forecast
    - Other runtime params from Airflow
    """
    from vprc import read_vvp

    ds = read_vvp(vvp_file, {
        'freezing_level_m': nwp_data['freezing_level'],
    })

    # ... processing logic
    return output_path
```

## Environment Variables

- `VPRC_RADAR_CONFIG`: Path to custom TOML configuration file
  - If set, loads this file instead of package default
  - Useful for testing or alternative radar networks
  - File must exist or `FileNotFoundError` is raised

## Radar Name Mapping

The parser automatically maps full radar names to three-letter codes:

- `KANKAANPAA` → `KAN`
- `VANTAA` → `VAN`
- `IKAALINEN` → `IKA`
- etc.

If the full name is not recognized, uses the first 3 characters as the code.

## Design Rationale

This three-tier precedence system provides:

1. **Development flexibility**: Override any field for testing
2. **Operational deployment**: Static config in container + dynamic NWP data
3. **Sensible defaults**: Works out-of-the-box for FMI radar network
4. **Traceability**: Package-shipped TOML matches Perl reference implementation
