#!/usr/bin/env python3
"""Interactive script for examining VPR processing results.

Run with: python -i src/scripts/examine_vpr.py [path_to_vvp_file]

Or from IPython:
    %run src/scripts/examine_vpr.py

Key variables available after running:
    ds          - xarray Dataset with corrected_dbz after all processing
    result      - ProcessedProfile with all processing results
    classification - Profile layer classification
    bright_band - Bright band detection results
    vpr_correction - VPR correction factors (if computed)

Individual processing steps are also available:
    ds_raw      - Raw dataset before any processing
    ds_clutter  - After ground clutter removal
    ds_smooth   - After spike smoothing
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Ensure package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vprc import (
    process_vvp,
    read_vvp,
    ProcessedProfile,
)
from vprc.clutter import remove_ground_clutter
from vprc.smoothing import smooth_spikes
from vprc.classification import classify_profile
from vprc.bright_band import detect_bright_band
from vprc.vpr_correction import compute_vpr_correction

# Default sample file
DEFAULT_VVP = Path(__file__).resolve().parents[2] / "tests/data/202511071400_VIH.VVP_40.txt"
#DEFAULT_VVP = Path(__file__).resolve().parents[2] / "tests/data/202508241100_KAN.VVP_40.txt"


def plot_profile(ds, ds_raw=None, bright_band=None, title: str | None = None) -> plt.Figure:
    """Plot dBZ and sample count profiles against height.

    Args:
        ds: xarray Dataset with 'corrected_dbz' and 'sample_count'
        ds_raw: Optional raw dataset to show original dBZ profile
        bright_band: Optional BrightBandResult to mark on the plot
        title: Optional plot title

    Returns:
        matplotlib Figure object
    """
    fig, ax1 = plt.subplots(figsize=(8, 10))

    height = ds["height"].values
    dbz = ds["corrected_dbz"].values
    count = ds["sample_count"].values

    # Plot raw dBZ if provided
    if ds_raw is not None:
        raw_dbz = ds_raw["corrected_dbz"].values
        ax1.plot(raw_dbz, height, color="tab:blue", linewidth=1, alpha=0.4, label="dBZ (raw)")

    # Plot dBZ on primary x-axis
    color_dbz = "tab:blue"
    ax1.plot(dbz, height, color=color_dbz, linewidth=2, label="dBZ")
    ax1.set_xlabel("Reflectivity (dBZ)", color=color_dbz)
    ax1.set_ylabel("Height (m)")
    ax1.tick_params(axis="x", labelcolor=color_dbz)
    ax1.set_xlim(-50, 50)
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # Mark bright band if detected
    if bright_band is not None and bright_band.detected:
        ax1.axhline(y=bright_band.peak_height, color="tab:red", linestyle="-", linewidth=2, label="BB peak")
        ax1.axhline(y=bright_band.top_height, color="tab:red", linestyle="--", alpha=0.7, label="BB top")
        ax1.axhline(y=bright_band.bottom_height, color="tab:red", linestyle="--", alpha=0.7, label="BB bottom")
        # Shade the bright band region
        ax1.axhspan(bright_band.bottom_height, bright_band.top_height, alpha=0.1, color="tab:red")

    # Plot sample count on secondary x-axis
    ax2 = ax1.twiny()
    color_count = "tab:orange"
    ax2.plot(count, height, color=color_count, linewidth=2, label="count")
    ax2.set_xlabel("Sample count", color=color_count)
    ax2.tick_params(axis="x", labelcolor=color_count)

    # Grid and title
    ax1.grid(True, alpha=0.3)
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig


def _plot_correction_line(ax, data, label, **kwargs):
    """Helper to plot a correction line.

    Args:
        ax: Matplotlib axis
        data: xarray DataArray with correction data
        label: Legend label
        **kwargs: Additional plot kwargs (linewidth, linestyle, color, etc.)
    """
    range_km = data["range_km"].values
    correction_db = data.values
    ax.plot(range_km, correction_db, label=label, **kwargs)


def plot_vpr_correction(vpr_correction, title: str | None = None) -> plt.Figure:
    """Plot VPR correction factors vs range for CAPPI heights.

    Args:
        vpr_correction: VPRCorrectionResult object with corrections
        title: Optional plot title

    Returns:
        matplotlib Figure object
    """
    if vpr_correction is None or not vpr_correction.usable:
        print("No VPR correction to plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    corr = vpr_correction.corrections
    if "cappi_correction_db" not in corr:
        print("No cappi_correction_db in corrections dataset")
        return None

    # Cappi 1000 orange color
    color_1000 = "tab:orange"

    # Plot correction for each CAPPI height
    cappi_heights = corr["cappi_height"].values
    for height in cappi_heights:
        data = corr["cappi_correction_db"].sel(cappi_height=height)
        _plot_correction_line(ax, data, f"CAPPI {height} m", linewidth=2)

    # Plot climatological correction for CAPPI 1000 if available
    if "cappi_clim_correction_db" in corr:
        data = corr["cappi_clim_correction_db"].sel(cappi_height=1000)
        _plot_correction_line(ax, data, "Climatology 1000 m",
                             linewidth=2, linestyle=":", color=color_1000)

    # Plot blended correction for CAPPI 1000 if available
    if "cappi_blended_correction_db" in corr:
        data = corr["cappi_blended_correction_db"].sel(cappi_height=1000)
        _plot_correction_line(ax, data, "Blended 1000 m",
                             linewidth=2, linestyle="--", color=color_1000)

    # Plot correction for lowest elevation angle if available
    if "elev_correction_db" in corr:
        elev_angles = corr["elevation"].values
        if len(elev_angles) > 0:
            lowest_elev = elev_angles[0]  # Assuming sorted, lowest first
            data = corr["elev_correction_db"].sel(elevation=lowest_elev)
            _plot_correction_line(ax, data, f"Elevation {lowest_elev:.1f}°", linewidth=2)

    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Correction (dB)")
    ax.set_title(title or "VPR Correction Factors")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Add ground reflectivity info to title or legend
    z_ground = vpr_correction.z_ground_dbz
    ax.text(0.02, 0.98, f"Ground reflectivity: {z_ground:.1f} dBZ",
            transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    return fig


def load_vvp(path: str | Path | None = None) -> ProcessedProfile:
    """Load and process a VVP file.

    Args:
        path: Path to VVP file. Uses default sample if None.

    Returns:
        ProcessedProfile with all results.
    """
    if path is None:
        path = DEFAULT_VVP
    path = Path(path)
    print(f"Loading: {path}")
    return process_vvp(path)


def load_step_by_step(path: str | Path | None = None) -> dict:
    """Load VVP file with intermediate results at each processing step.

    Useful for debugging or understanding the processing pipeline.

    Args:
        path: Path to VVP file. Uses default sample if None.

    Returns:
        Dictionary with datasets at each processing step.
    """
    if path is None:
        path = DEFAULT_VVP
    path = Path(path)
    print(f"Loading step-by-step: {path}")

    ds_raw = read_vvp(path)
    ds_clutter = remove_ground_clutter(ds_raw)
    ds_smooth = smooth_spikes(ds_clutter)
    classification = classify_profile(ds_smooth)

    layer_top = None
    if classification.lowest_layer is not None:
        layer_top = classification.lowest_layer.top_height

    bright_band = detect_bright_band(ds_smooth, layer_top=layer_top)
    freezing_level_m = bright_band.top_height if bright_band.detected else None

    vpr_correction = None
    if classification.usable_for_vpr:
        vpr_correction = compute_vpr_correction(ds_smooth, freezing_level_m=freezing_level_m)
    return {
        "ds_raw": ds_raw,
        "ds_clutter": ds_clutter,
        "ds_smooth": ds_smooth,
        "ds": ds_smooth,  # alias for final dataset
        "classification": classification,
        "bright_band": bright_band,
        "vpr_correction": vpr_correction,
    }


if __name__ == "__main__":
    plt.ion()  # Interactive mode on

    # Get VVP file path from command line or use default
    vvp_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VVP

    # Load with step-by-step results for maximum visibility
    steps = load_step_by_step(vvp_path)

    # Unpack into module namespace for interactive use
    ds_raw = steps["ds_raw"]
    ds_clutter = steps["ds_clutter"]
    ds_smooth = steps["ds_smooth"]
    ds = steps["ds"]
    classification = steps["classification"]
    bright_band = steps["bright_band"]
    vpr_correction = steps["vpr_correction"]

    # Also run the full pipeline for comparison
    result = process_vvp(vvp_path)

    # Print summary
    print("\n" + "=" * 60)
    print("VPR Processing Results")
    print("=" * 60)
    print(f"\nProfile type: {classification.profile_type}")
    print(f"Usable for VPR: {classification.usable_for_vpr}")

    if classification.layers:
        print(f"\nLayers ({len(classification.layers)}):")
        for i, layer in enumerate(classification.layers, 1):
            print(f"  {i}. {layer.layer_type.name}: {layer.bottom_height}–{layer.top_height} m")

    print(f"\nBright band detected: {bright_band.detected}")
    if bright_band.detected:
        print(f"  Peak height: {bright_band.peak_height} m")
        print(f"  Top: {bright_band.top_height} m, Bottom: {bright_band.bottom_height} m")
        print(f"  Peak dBZ: {bright_band.peak_dbz:.1f}")

    if vpr_correction is not None:
        print(f"\nVPR correction computed: usable={vpr_correction.usable}")
        print(f"  Ground reflectivity: {vpr_correction.z_ground_dbz:.1f} dBZ")
    else:
        print("\nVPR correction: not computed (profile not usable)")

    print("\n" + "-" * 60)
    print("Available variables:")
    print("  ds          - Final processed dataset")
    print("  ds_raw      - Raw input dataset")
    print("  ds_clutter  - After ground clutter removal")
    print("  ds_smooth   - After spike smoothing")
    print("  result      - ProcessedProfile object")
    print("  classification, bright_band, vpr_correction")
    print("  fig         - Profile plot figure")
    print("  fig_corr    - VPR correction plot figure (if available)")
    print("-" * 60 + "\n")

    # Create profile plot
    fig = plot_profile(ds, ds_raw=ds_raw, bright_band=bright_band, title=str(vvp_path.name))

    # Create VPR correction plot if available
    fig_corr = None
    if vpr_correction is not None:
        fig_corr = plot_vpr_correction(vpr_correction, title=f"VPR Correction - {vvp_path.name}")
