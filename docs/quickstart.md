# Quick Start

## Basic Usage

```python
from vprc import process_vvp

# Process a VVP profile with freezing level from NWP
result = process_vvp(
    "202508241100_KAN.VVP_40.txt",
    freezing_level_m=2500,
)

# Check if profile is usable for VPR correction
if result.usable_for_vpr:
    print(f"Profile type: {result.classification.profile_type}")

    # Bright band detection
    if result.bright_band.detected:
        bb = result.bright_band
        print(f"Bright band: {bb.bottom_height}–{bb.top_height} m")

    # VPR correction factors by range
    if result.vpr_correction:
        corr = result.vpr_correction.corrections
        value = corr.cappi_blended_correction_db.sel(range_km=100, cappi_height=500)
        print(f"Correction at 100 km: {float(value):.2f} dB")
```

## Accessing Radar Configuration

Use the config API to retrieve radar metadata programmatically:

```python
from vprc import get_radar_metadata, get_radar_coords, process_vvp

# Get radar coordinates for compositing
lat, lon = get_radar_coords('KAN')

# Get all metadata
meta = get_radar_metadata('VIH')
print(meta['antenna_height_m'])  # 181

# Override runtime parameters (e.g., freezing level from NWP)
result = process_vvp(
    "202508241100_KAN.VVP_40.txt",
    freezing_level_m=2000,
)
```

See [configuration.md](configuration.md) for details on TOML structure and precedence.

Freezing levels passed to `process_vvp()` or `read_vvp()` are meters above sea
level. Dataset heights and the low-level correction API use meters above the
antenna. CAPPI target heights are above sea level.

## Temporal Averaging

```python
from vprc import average_corrections

# Results must have identical correction variables and range/height coordinates.
averaged = average_corrections(correction_results)

# Opt in to pystycappi_ka.pl's oldest-to-newest rank weights.
legacy_averaged = average_corrections(correction_results, weighting="legacy")
```

The default `"elapsed"` weighting retains actual-time decay with oldest weight
0.2; `"legacy"` uses `(j + 1) / n` after sorting by timestamp. Both average the
already-blended corrections in dB, preserve climatological products, and average
quality weights for compositing. Non-finite values and incompatible grids are
errors. Select the desired observation window in the calling workflow; the
legacy operational window was six hours.

## Low-Level API

For finer control, use the individual processing functions:

```python
from vprc import read_vvp
from vprc.clutter import remove_ground_clutter
from vprc.smoothing import smooth_spikes
from vprc.classification import classify_profile
from vprc.bright_band import detect_bright_band
from vprc.vpr_correction import compute_vpr_correction

# Step through the pipeline manually
ds = read_vvp("profile.txt", radar_metadata={"freezing_level_m": 2500})
ds = remove_ground_clutter(ds)
ds = smooth_spikes(ds)
classification = classify_profile(ds)
bright_band = detect_bright_band(ds)

if classification.usable_for_vpr:
    vpr_result = compute_vpr_correction(
        ds, freezing_level_m=ds.attrs["freezing_level_m"], bb_result=bright_band
    )
```

`read_vvp()` and `process_vvp()` raise `vprc.NoUsableProfileError`, a
`ValueError` subclass, for valid files without a usable profile: header-only
products and profiles with no levels at or above the radar's lowest profile
level. Catch it to fall back to climatology without treating the file as a
processing failure. Other exceptions mean invalid input or configuration.

The composite creation command exits with an error for unreadable/corrupt inputs,
files without a usable profile, missing radar coordinates, or processing
failures. These failures must not be interpreted as valid no-precipitation
products.

## Airflow Integration

```python
@task.docker(image="quay.io/fmi/vprc:v0.2.0")
def correct_vpr(vvp_file: str, radar_config: dict) -> dict:
    from vprc import process_vvp
    result = process_vvp(vvp_file, **radar_config)
    return {
        "usable": result.usable_for_vpr,
        "profile_type": str(result.classification.profile_type),
    }
```
