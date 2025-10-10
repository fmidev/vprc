# Copilot Instructions for vprc

## Project Overview

This is a Python reimplementation of the **Koistinen & Pohjola VPR (Vertical Profile Reflectivity) correction algorithm** for weather radar data. The original Perl codebase (circa 2003) is being modernized for use in Apache Airflow workflows at FMI (Finnish Meteorological Institute).

**Key insight**: This is a reference implementation project. The legacy Perl scripts (`allprof_prodx2.pl`, `pystycappi.pl`, `pystycappi_ka.pl`) remain in the repository as the authoritative algorithm specification. Do not modify or delete these files.

**Core values**: Modern tools and standards, code readability and maintainability, PEP 8, reuse over reimplementation. Leverage `numpy`, `xarray`, `wradlib`, and other established libraries for mathematical and radar-specific operations instead of reimplementing algorithms. **Do it the pythonic way.**

## Architecture

### Data Flow Pipeline

1. **Input**: IRIS VVP (Vertical Velocity Profile) prodx files from radar stations
   - Format: Text files with header + tabular vertical profile data
   - Example: `tests/data/202508241100_KAN.VVP_40.txt` (Kankaanpää radar)

2. **Parsing** (`src/vprc/io.py`):
   - `read_vvp()` → Returns `xarray.Dataset`

3. **Processing** (not yet implemented):
   - Bright band detection and correction
   - Ground clutter removal
   - Profile quality weighting
   - Spike smoothing
   - VPR correction factor calculation

4. **Output**: Corrected radar reflectivity profiles for operational use

### Radar-Specific Metadata

Each radar station has unique characteristics (antenna height, beam angles, horizon obstructions). Pass this metadata via `radar_metadata` dict to `read_vvp()`:

```python
radar_meta = {
    'antenna_height_km': 0.174,      # Antenna elevation above sea level
    'lowest_level_offset_m': 126,    # Offset from nearest profile level
    'freezing_level_m': 2000,        # From MEPS/sounding data
}
```

**Configuration sources** (in priority order):
1. Runtime parameters from Airflow (operational deployment)
2. Static TOML files (development/testing)
3. Default values from reference implementation (see `allprof_prodx2.pl` lines 32-50)

Ship a `radar_defaults.toml` with the package containing canonical radar configurations.

## Airflow Integration

This package will be deployed as a containerized service in FMI's Airflow radar production system. The integration pattern:

- **Deployment**: Docker container with this package installed
- **Airflow tasks**: Use `@task.docker` decorator to invoke Python API
- **Configuration**: Radar metadata and algorithm parameters from Airflow (static TOML files initially)
- **No DAGs in this repo**: Workflow orchestration lives in the separate Airflow radar production repository

Example Airflow task (external to this repo):
```python
@task.docker(image="fmi/vprc:latest")
def correct_vpr(vvp_file: str, radar_config: dict) -> str:
    from vprc import process_vvp
    return process_vvp(vvp_file, radar_config)
```

## Key Conventions

### Data Representation

- **Heights**: Integer meters above antenna (not sea level), in 200m steps: 100, 300, 500, ...
- **dBZ values**: Use `lin_dbz` column (linear reflectivity), not `log_dbz`
- **Missing data**: Represented as `-45` dBZ (the MDS threshold in legacy code)
- **xarray coordinate**: Use `height` dimension for all vertical profiles

### Testing

- **Run tests**: `pytest tests/` (requires sample data in `tests/data/`)
- **Test structure**: One test file per module (`test_io.py` ↔ `src/vprc/io.py`)
- **Use fixtures**: Test data files should be in `tests/data/` and checked into git
- **Validation approach**: Compare against legacy Perl output when implementing algorithms

### Algorithm Implementation Strategy

**Development sequence** (logical order for testing against Perl reference):

1. **Ground clutter removal** (`allprof_prodx2.pl` lines 393-488) - Simplest, enables early validation
2. **Spike smoothing** (lines 490-686) - Independent preprocessing step
3. **Profile quality weighting** (lines 1250+) - Needed for correction calculation
4. **Bright band detection** (lines 897-1166) - Complex but well-defined
5. **VPR correction calculation** (`pystycappi.pl`) - Final integration

**Implementation guidelines**:

1. **Consult Perl for algorithm logic**: Understand *what* to compute from heavily-commented Finnish code
2. **Implement pythonically**: Use numpy operations, xarray methods, and wradlib functions instead of literal translation
3. **Match numerical results**: Validate against Perl output, but code structure can differ significantly
4. **Document mapping**: Reference Perl line numbers in docstrings for traceability

**Key algorithm constants** (from `allprof_prodx2.pl`):
```python
MDS = -45              # Minimum detectable signal (dBZ)
STEP = 200             # Vertical resolution (m)
MK_THRESHOLD = -0.005  # Gradient threshold: -1 dBZ/200m
DBZ_KYNNYS1 = 4        # Spike detection thresholds
```

**Profile classification**: `Prec.` (precipitation), `As` (altostratus), `CAE` (clear air echo), `Clutter`

### Dependencies

- **Core data**: `pandas`, `xarray`, `numpy` - array operations and data structures
- **Radar domain**: `wradlib` - radar-specific algorithms and transformations
- **Config**: `tomllib` - TOML configuration parsing (stdlib in Python 3.11+)
- **Build system**: `hatch` with `hatch-vcs` for versioning
- **Python support**: 3.12+

## Development Workflow

### Adding New Processing Functions

When implementing algorithm components (e.g., bright band detection):

1. **Study Perl reference**: Understand algorithm logic from `allprof_prodx2.pl` (comments in Finnish)
2. **Research existing solutions**: Check if `wradlib`, `numpy`, or other packages provide relevant functions
3. **Create module**: Add to `src/vprc/` (e.g., `bright_band.py`) using pythonic patterns
4. **Write tests**: Add to `tests/test_<module>.py` with validation against Perl output
5. **Document traceability**: Reference Perl line numbers in docstrings

Example function signature:
```python
def detect_bright_band(ds: xr.Dataset) -> xr.Dataset:
    """Based on algorithm from allprof_prodx2.pl lines 897-1166."""
    dbz_gradient = ds['corrected_dbz'].differentiate('height')
    # ... pythonic implementation using numpy/xarray operations
```

## Common Pitfalls

- **Don't assume sea level heights**: Input files use heights relative to antenna
- **Column name confusion**: Perl uses `$dbz`, Python uses `lin_dbz` (avoid `log_dbz` unless logarithmic)
- **Zero vs. missing**: Legacy code uses `-45` for missing, not `NaN` or `0`
- **Radar codes**: Three-letter codes (KAN, VAN, IKA) map to full names (see `allprof_prodx2.pl` line 32+)

## File Organization

- `src/vprc/io.py` - VVP file parsing (complete)
- `src/vprc/*.py` - Algorithm modules (to be added: `clutter.py`, `smoothing.py`, `bright_band.py`, etc.)
- `src/vprc/radar_defaults.toml` - Canonical radar configurations (ship with package)
- `tests/test_*.py` - Test modules (mirror `src/vprc/` structure)
- `tests/data/` - Sample VVP files and expected Perl outputs for validation
- `*.pl` - Legacy Perl scripts (read-only algorithm reference, do not modify)
- `*.tcsh` - Legacy workflow scripts (reference only, shows legacy operational context)
