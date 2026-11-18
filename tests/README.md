# Test Suite for vprc

This directory contains the test suite for the VPR correction project.

## Structure

```
tests/
├── README.md              # This file
├── __init__.py            # Test package initialization
├── legacy_parser.py       # Utility for parsing legacy Perl outputs
├── test_io.py             # VVP file parsing
├── test_clutter.py        # Ground clutter removal
├── test_smoothing.py      # Spike smoothing
├── test_classification.py # Profile classification
├── test_bright_band.py    # Bright band detection
├── test_vpr_correction.py # VPR correction calculation
├── test_legacy_comparison.py  # Validation against Perl output
└── data/                  # Test data files
    ├── 202508241100_KAN.VVP_40.txt      # Sample VVP (Kankaanpää)
    ├── 202508241100_KAN.VVP_40.profile  # Legacy profile output
    ├── 202508241100_KAN.VVP_40.cor      # Legacy correction output
    ├── 202511071400_VIH.VVP_40.txt      # Sample VVP (Vihti)
    ├── 202511071400_VIH.VVP_40.profile  # Legacy profile output
    ├── 202611112000_KES.VVP_40.txt      # VVP lacking 500/1300/1700 m ASL
    ├── 202611112000_KES.VVP_40.profile  # Legacy profile output
    ├── variant_KAN_level2_low_zcnt.VVP_40.*  # 2025 KAN, ZCnt 20 at 500 m ASL
    └── variant_KAN_no_lowest_level.VVP_40.*  # 2025 KAN without 300 m ASL
```

## Running Tests

Run all tests:
```bash
uv run pytest tests/
```

Run tests for a specific module:
```bash
uv run pytest tests/test_io.py
```

Run tests with verbose output:
```bash
uv run pytest tests/ -v
```

Run tests with coverage:
```bash
uv run pytest tests/ --cov=src/vprc --cov-report=term-missing
```

Run a specific test class:
```bash
uv run pytest tests/test_io.py::TestParseVVPFile -v
```

Run a specific test:
```bash
uv run pytest tests/test_io.py::TestParseVVPFile::test_parse_sample_file -v
```

## Test Organization

Each test file corresponds to a module in `src/vprc/`:

| Test File | Module | Description |
|-----------|--------|-------------|
| `test_io.py` | `io.py` | VVP file parsing and xarray conversion |
| `test_clutter.py` | `clutter.py` | Ground clutter removal |
| `test_smoothing.py` | `smoothing.py` | Spike smoothing algorithms |
| `test_classification.py` | `classification.py` | Profile layer classification |
| `test_bright_band.py` | `bright_band.py` | Bright band detection |
| `test_vpr_correction.py` | `vpr_correction.py` | VPR correction calculation |
| `test_legacy_comparison.py` | – | End-to-end validation against Perl output |
| `test_legacy_parser.py` | `legacy_parser.py` | Reference fields, multiple layers and BB candidate sections |
| `test_pipeline.py` | `__init__.py` | Public pipeline metadata and stage integration |
| `test_temporal.py` | `temporal.py` | Weighting, product preservation and input compatibility |
| `test_temporal_e2e.py` | `temporal.py` | Numerical temporal comparison against `.corave` |
| `test_composite.py` | `composite.py` | Spatial weights and temporal-to-composite integration |
| `test_io_export.py` | `io/geotiff.py` | COG output |
| `test_create_composite.py` | `scripts/create_composite.py` | Processing errors versus valid climatology fallback |

## Test Data

Test data files in `tests/data/`:

### Input Files

- `202508241100_KAN.VVP_40.txt` – Kankaanpää radar, 2025-08-24 11:00 UTC
- `202511071400_VIH.VVP_40.txt` – Vihti radar, 2025-11-07 14:00 UTC
- `202611112000_KES.VVP_40.txt` – Kesälahti radar, 2026-11-11 20:00 UTC; the
  file lacks the 500, 1300 and 1700 m ASL levels
- `variant_KAN_*.VVP_40.txt` – the Kankaanpää input with one change each:
  ZCnt 20 at 500 m ASL (`level2_low_zcnt`) or the 300 m ASL row removed
  (`no_lowest_level`). Their `.profile` and `.cor` references were produced by
  running the unmodified `allprof_prodx2.pl` and `pystycappi.pl` on them
  without a HIRLAM freezing level.

### Legacy Reference Outputs

Used by `test_legacy_comparison.py` to validate against the original Perl implementation:

- `*.profile` – Processed profile output from `allprof_prodx2.pl`
- `*.cor` – Correction factors from `pystycappi.pl`

### Reference Validation

The temporal numerical test feeds the checked-in instantaneous `.cor` values
through `average_corrections(weighting="legacy")` and compares all six products
at every reference range. Its 0.051 dB tolerance accounts for the reference
output's one-decimal rounding. Default elapsed-time weights are tested
separately; they intentionally differ from legacy rank weights.

Synthetic profiles must distinguish wind `sample_count` from reflectivity
`zcount`, and raw `lin_dbz` from working `corrected_dbz`. Keeping those fields
identical in every test hides important algorithm errors. Assert detection,
classification, and numerical corrections directly, not just metadata,
plausible ranges, or trend direction.

The legacy sources are optional, read-only local references. Tests should use
checked-in outputs or explicit analytical expectations and must not require
modifying or running scripts in the legacy working directory.

## Adding New Tests

1. Create `test_<module_name>.py` in `tests/`
2. Organize tests into classes by function/feature
3. Use descriptive names: `test_<feature>_<scenario>`
4. Place test data files in `tests/data/`
5. Use pytest fixtures for shared setup

## Dependencies

Test dependencies are in the `dev` dependency group of `pyproject.toml`:

```bash
uv sync
uv run pytest tests/
```
