# Copilot Instructions for vprc

## Project Overview

This is a Python reimplementation of the **Koistinen & Pohjola VPR (Vertical Profile Reflectivity) correction algorithm** for weather radar data. The original Perl codebase (circa 2003) is being modernized for use in Apache Airflow workflows at FMI (Finnish Meteorological Institute).

**Key insight**: This is a modernization project. The legacy Perl scripts (`allprof_prodx2.pl`, `pystycappi.pl`, `pystycappi_ka.pl`) remain in the `legacy/` subdirectory as the authoritative algorithm specification. In the directory, there is also an input IRIS prodx VVP profile and output files. Do not modify or delete any of these files.

**Core values**: Modern tools and standards, code readability and maintainability, reuse over reimplementation. Where applicable, leverage `scipy`, `xarray`, `wradlib`, and other established libraries for mathematical and radar-specific operations instead of reimplementing algorithms. **Do it the pythonic way.**

## Architecture

### Data Flow Pipeline

1. **Input**: IRIS VVP (Velocity Volume Processing) prodx files from radar stations
   - Format: Text files with header + tabular vertical profile data
   - Example: `tests/data/202508241100_KAN.VVP_40.txt` (Kankaanpää radar)

2. **Parsing** ([src/vprc/io/](src/vprc/io/)):
   - `read_vvp()` → Returns `xarray.Dataset`

3. **Processing** (modular pipeline):
   - Ground clutter removal ([src/vprc/clutter.py](src/vprc/clutter.py))
   - Spike smoothing ([src/vprc/smoothing.py](src/vprc/smoothing.py))
   - Profile classification ([src/vprc/classification.py](src/vprc/classification.py))
   - Bright band detection ([src/vprc/bright_band.py](src/vprc/bright_band.py))
   - VPR correction factor calculation ([src/vprc/vpr_correction.py](src/vprc/vpr_correction.py))
   - Temporal averaging ([src/vprc/temporal.py](src/vprc/temporal.py))
   - Climatology blending ([src/vprc/climatology.py](src/vprc/climatology.py))
   - Compositing VPR corrections ([src/vprc/composite.py](src/vprc/composite.py))

4. **Output**: Corrected radar reflectivity profiles and VPR correction factors for operational use

### Radar-Specific Metadata

Each radar station has unique characteristics (antenna height, beam angles, horizon obstructions). Pass this metadata via `radar_metadata` dict to `read_vvp()`:

```python
radar_meta = {
    'antenna_height_m': 174,      # Antenna elevation above sea level
    'lowest_level_offset_m': 126,    # Offset from nearest profile level
    'freezing_level_m': 2000,        # From NWP data
}
```

**Configuration sources** (in priority order):
1. Runtime parameters from Airflow (operational deployment)
2. Static TOML files (development/testing)
3. Default values from reference implementation (see beginning of `allprof_prodx2.pl`)

Ship a `radar_defaults.toml` with the package containing canonical radar configurations.

## Airflow Integration

This package is deployed as a containerized service in FMI's Airflow radar production system. The integration pattern:

- **Deployment**: Docker container with this package installed (built via [Containerfile](Containerfile), image `quay.io/fmi/vprc`)
- **Airflow tasks**: Use `@task.docker` decorator to invoke Python API
- **Configuration**: Radar metadata from TOML files ([src/vprc/radar_defaults.toml](src/vprc/radar_defaults.toml)), runtime parameters from Airflow
- **No DAGs in this repo**: Workflow orchestration lives in the separate Airflow radar production repository

Example Airflow task (external to this repo):
```python
@task.docker(image="quay.io/fmi/vprc:latest")
def correct_vpr(vvp_file: str, freezing_level_m: float) -> dict:
    from vprc import process_vvp
    result = process_vvp(vvp_file, freezing_level_m=freezing_level_m)
    return {
        "usable": result.usable_for_vpr,
        "profile_type": result.classification.profile_type,
        "bright_band_detected": result.bright_band.detected,
    }
```

## Key Conventions

### Data Representation

- **Heights**: Integer meters above antenna (converted from ASL), in 200m steps, e.g., 119, 319, 519, ...
- **dBZ values**: Use `lin_dbz` column (reflectivity in dBZ), not `log_dbz`
- **Missing data**: Represented as `-45` dBZ (the MDS threshold in legacy code)
- **xarray coordinate**: Use `height` dimension for all vertical profiles

### Testing

- **Run tests**: `pytest tests/` (requires sample data in [tests/data/](tests/data/))
- **Test structure**: One test file per module ([tests/test_clutter.py](tests/test_clutter.py) ↔ [src/vprc/clutter.py](src/vprc/clutter.py))
- **Use fixtures**: Test data files in [tests/data/](tests/data/) (checked into git)
- **Validation approach**: Compare against legacy Perl output ([tests/test_legacy_comparison.py](tests/test_legacy_comparison.py))
- **Documentation**: See [tests/README.md](tests/README.md) for details on test structure and coverage

### Style
- Follow Black formatting
- Naming, comments, etc. in English
- Mention corresponding Perl names for key variables in comments/docstrings if helpful
- It's better to briefly quote legacy code than to refer to line numbers
- Use `logging` module for debug/info messages
- Type hinting for all functions
- Succinct, to the point documentation
- Avoid repeating bad practices from legacy code

### Algorithm Implementation Strategy

**Implementation guidelines**:

1. **Consult Perl for algorithm logic**: Understand *what* to compute from heavily-commented Finnish code
2. **Implement pythonically**: Use numpy operations, xarray methods, and wradlib functions instead of literal translation
3. **Match numerical results**: Validate against Perl output, but code structure can differ significantly
4. **No hardcoding**: Use configuration files and input parameters. The radar network will evolve over time.

### Profile types

`Prec.` (precipitation), `As` (altostratus), `CAE` (clear air echo), `Clutter`

### Dependencies

- **Core data**: `pandas`, `xarray`, `numpy` - array operations and data structures
- **Radar domain**: `wradlib` - radar-specific algorithms and transformations
- **Build system**: `hatch` with `hatch-vcs` for versioning
- **Python support**: 3.12+

## Development Workflow

### Extending the Algorithm

When implementing new features or algorithm improvements:

1. **Study Perl reference**: Understand algorithm logic from [legacy/allprof_prodx2.pl](legacy/allprof_prodx2.pl) (comments in Finnish)
2. **Research existing solutions**: Check if `wradlib`, `numpy`, or other packages provide relevant functions
3. **Create/modify module**: Update existing module in [src/vprc/](src/vprc/) using pythonic patterns
4. **Write/update tests**: Add to corresponding [tests/test_<module>.py](tests/) with validation against Perl output
5. **Update documentation**: On major changes, update [docs/](docs/) and this instruction file

Example function signature pattern:
```python
def detect_bright_band(ds: xr.Dataset, freezing_level_m: float) -> BrightBandResult:
    """Detect melting layer using gradient analysis.

    Based on 'Bright Band' algorithm in allprof_prodx2.pl.
    """
    # ... pythonic implementation using numpy/xarray operations
```

## File Organization

What's currently implemented:

**Core implementation** ([src/vprc/](src/vprc/)):
- [__init__.py](src/vprc/__init__.py) - Main API and pipeline orchestration
- [io/](src/vprc/io/) - I/O module (VVP parsing and GeoTIFF export)
  - [vvp.py](src/vprc/io/vvp.py) - VVP file parsing
  - [geotiff.py](src/vprc/io/geotiff.py) - GeoTIFF export (Cloud Optimized GeoTIFFs)
  - [__init__.py](src/vprc/io/__init__.py) - Public API shortcuts
- [clutter.py](src/vprc/clutter.py) - Ground clutter removal
- [smoothing.py](src/vprc/smoothing.py) - Spike smoothing
- [classification.py](src/vprc/classification.py) - Profile classification
- [bright_band.py](src/vprc/bright_band.py) - Bright band detection
- [vpr_correction.py](src/vprc/vpr_correction.py) - VPR correction calculation
- [temporal.py](src/vprc/temporal.py) - Temporal averaging
- [composite.py](src/vprc/composite.py) - Compositing VPR corrections
- [constants.py](src/vprc/constants.py) - Algorithm constants (MDS, STEP, etc.)
- [radar_defaults.toml](src/vprc/radar_defaults.toml) - Canonical radar configurations

**Testing** ([tests/](tests/)):
- [test_*.py](tests/) - Test modules (mirror [src/vprc/](src/vprc/) structure)
- [test_legacy_comparison.py](tests/test_legacy_comparison.py) - End-to-end validation against Perl
- [test_temporal_e2e.py](tests/test_temporal_e2e.py) - Temporal averaging integration test
- [legacy_parser.py](tests/legacy_parser.py) - Utility for parsing Perl output
- [data/](tests/data/) - Sample VVP files and expected Perl outputs

**Documentation**:
- [docs/quickstart.md](docs/quickstart.md) - Usage examples
- [docs/introduction.md](docs/introduction.md) - Algorithm overview
- [docs/configuration.md](docs/configuration.md) - TOML configuration guide
- [tests/README.md](tests/README.md) - Test suite documentation

**Legacy reference** ([legacy/](legacy/)):
- [allprof_prodx2.pl](legacy/allprof_prodx2.pl) - Main algorithm (read-only reference)
- [pystycappi.pl](legacy/pystycappi.pl), [pystycappi_ka.pl](legacy/pystycappi_ka.pl) - CAPPI correction scripts
- `*.tcsh` - Legacy workflow scripts (operational context reference)
- `*.txt`, `*.profile`, `*.cor` - Sample input/output data

**Development tools** not in version control ([local/](local/)):
- [scripts/](local/scripts/) - Development utilities
- [data/](local/data/) - Additional test data
- [output/](local/output/) - Output files

## Common Pitfalls

- **Column name confusion**: Perl uses `$dbz`, Python uses `lin_dbz` (avoid `log_dbz` unless logarithmic). `lin_dbz` is not to be confused with linear reflectivity Z (in mm^6/m^3).
- **Zero vs. missing**: Legacy code uses `-45` for missing, not `NaN` or `0`
- **Radar codes**: Three-letter codes (KAN, VAN, IKA) map to full names (see `allprof_prodx2.pl` line 32+)
