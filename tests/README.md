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
    └── 202511071400_VIH.VVP_40.profile  # Legacy profile output
```

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run tests for a specific module:
```bash
pytest tests/test_io.py
```

Run tests with verbose output:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ --cov=src/vprc --cov-report=term-missing
```

Run a specific test class:
```bash
pytest tests/test_io.py::TestParseVVPFile -v
```

Run a specific test:
```bash
pytest tests/test_io.py::TestParseVVPFile::test_parse_sample_file -v
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

## Test Data

Test data files in `tests/data/`:

### Input Files

- `202508241100_KAN.VVP_40.txt` – Kankaanpää radar, 2025-08-24 11:00 UTC
- `202511071400_VIH.VVP_40.txt` – Vihti radar, 2025-11-07 14:00 UTC

### Legacy Reference Outputs

Used by `test_legacy_comparison.py` to validate against the original Perl implementation:

- `*.profile` – Processed profile output from `allprof_prodx2.pl`
- `*.cor` – Correction factors from `pystycappi.pl`

## Adding New Tests

1. Create `test_<module_name>.py` in `tests/`
2. Organize tests into classes by function/feature
3. Use descriptive names: `test_<feature>_<scenario>`
4. Place test data files in `tests/data/`
5. Use pytest fixtures for shared setup

## Dependencies

```bash
pip install pytest pytest-cov
```
