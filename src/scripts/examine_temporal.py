#!/usr/bin/env python3
"""Interactive script for examining temporal averaging results.

Run with: python -i src/scripts/examine_temporal.py

Or from IPython:
    %run src/scripts/examine_temporal.py

Key variables available after running:
    results     - List of ProcessedProfile objects for each timestep
    vpr_results - List of VPRCorrectionResult objects (non-None only)
    averaged    - VPRCorrectionResult after temporal averaging
    timestamps  - Timestamps of the profiles
    fig         - Comparison plot figure
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vprc import process_vvp, average_corrections
from vprc.vpr_correction import VPRCorrectionResult

# Default: Vimpeli test data (multiple timesteps)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "tests/data"
DEFAULT_PATTERN = "202208171*_VIM.VVP_40.txt"


def load_profiles(data_dir: Path | None = None, pattern: str | None = None) -> dict:
    """Load multiple VVP profiles for temporal averaging.

    Args:
        data_dir: Directory containing VVP files.
        pattern: Glob pattern for VVP files.

    Returns:
        Dictionary with results, vpr_results, timestamps, and averaged.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if pattern is None:
        pattern = DEFAULT_PATTERN

    vvp_files = sorted(data_dir.glob(pattern))
    if not vvp_files:
        raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")

    print(f"Loading {len(vvp_files)} profiles from {data_dir}")

    results = []
    for f in vvp_files:
        print(f"  {f.name}")
        results.append(process_vvp(f))

    # Extract VPR results with valid corrections
    vpr_results = [r.vpr_correction for r in results if r.vpr_correction is not None]
    timestamps = [r.vpr_correction.corrections.attrs.get("timestamp") for r in results if r.vpr_correction is not None]

    print(f"\n{len(vpr_results)} profiles usable for VPR correction")

    # Compute temporal average
    averaged = None
    if len(vpr_results) > 1:
        averaged = average_corrections(vpr_results)
        print(f"Temporal averaging complete (from {len(vpr_results)} profiles)")
    elif len(vpr_results) == 1:
        averaged = vpr_results[0]
        print("Only one profile - no averaging performed")

    return {
        "results": results,
        "vpr_results": vpr_results,
        "timestamps": timestamps,
        "averaged": averaged,
    }


def _format_timestamp(ts: str | None) -> str:
    """Format ISO timestamp to HH:MM for display."""
    if ts is None:
        return "?"
    # Handle both datetime objects and ISO strings
    if hasattr(ts, "strftime"):
        return ts.strftime("%H:%M")
    # ISO format: 2022-08-17T11:00:00
    try:
        return ts[11:16]  # Extract HH:MM
    except (IndexError, TypeError):
        return str(ts)[:5]


def plot_correction_comparison(
    vpr_results: list[VPRCorrectionResult],
    averaged: VPRCorrectionResult | None,
    cappi_height: int = 500,
    title: str | None = None,
) -> plt.Figure:
    """Plot individual and averaged VPR correction profiles.

    Args:
        vpr_results: Individual VPRCorrectionResult objects.
        averaged: Temporally averaged VPRCorrectionResult.
        cappi_height: CAPPI height to plot (m).
        title: Optional plot title.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Identify the newest (original) timestamp
    original_ts = averaged.corrections.attrs.get("timestamp") if averaged is not None else None

    # Plot individual profiles with timestamps in legend
    for vpr in vpr_results:
        corr = vpr.corrections
        if "cappi_correction_db" in corr:
            ts = corr.attrs.get("timestamp")
            label = _format_timestamp(ts)
            data = corr["cappi_correction_db"].sel(cappi_height=cappi_height)
            range_km = data["range_km"].values
            correction_db = data.values
            # Make the original (newest) profile thicker
            if ts == original_ts:
                ax.plot(range_km, correction_db, alpha=0.7, linewidth=2.5, label=label)
            else:
                ax.plot(range_km, correction_db, alpha=0.4, linewidth=1, label=label)

    # Plot averaged profile
    if averaged is not None and "cappi_correction_db" in averaged.corrections:
        avg_data = averaged.corrections["cappi_correction_db"].sel(cappi_height=cappi_height)
        range_km = avg_data["range_km"].values
        avg_correction_db = avg_data.values
        # Get the "original" (newest) timestamp for the averaged result
        avg_ts = averaged.corrections.attrs.get("timestamp")
        avg_label = f"Averaged ({_format_timestamp(avg_ts)})"
        ax.plot(range_km, avg_correction_db, color="black", linewidth=3, label=avg_label)

    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Correction (dB)")
    # Include original timestamp in title
    if title is None and averaged is not None:
        avg_ts = averaged.corrections.attrs.get("timestamp", "")
        radar_name = averaged.corrections.attrs.get("radar", "") + " "
        title = f"{radar_name}VPR Correction at CAPPI {cappi_height} m ({avg_ts[:16] if avg_ts else ''})"
    ax.set_title(title or f"VPR Correction at CAPPI {cappi_height} m")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_z_ground_comparison(
    vpr_results: list[VPRCorrectionResult],
    averaged: VPRCorrectionResult | None,
    title: str | None = None,
) -> plt.Figure:
    """Plot ground reflectivity values across profiles.

    Args:
        vpr_results: Individual VPRCorrectionResult objects.
        averaged: Temporally averaged VPRCorrectionResult.
        title: Optional plot title.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    z_grounds = [vpr.z_ground_dbz for vpr in vpr_results]
    timestamps = [_format_timestamp(vpr.corrections.attrs.get("timestamp")) for vpr in vpr_results]
    x = range(len(z_grounds))

    ax.bar(x, z_grounds, alpha=0.6, label="Individual profiles")
    ax.set_xticks(x)
    ax.set_xticklabels(timestamps, rotation=45, ha="right")

    if averaged is not None:
        ax.axhline(y=averaged.z_ground_dbz, color="red", linewidth=2, linestyle="--",
                   label=f"Averaged: {averaged.z_ground_dbz:.1f} dBZ")

    ax.set_xlabel("Time")
    ax.set_ylabel("Ground reflectivity (dBZ)")
    # Include original timestamp in title
    if title is None and averaged is not None:
        avg_ts = averaged.corrections.attrs.get("timestamp", "")
        title = f"Ground Reflectivity ({avg_ts[:16] if avg_ts else ''})"
    ax.set_title(title or "Ground Reflectivity Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    plt.ion()  # Interactive mode

    # Allow custom directory from command line
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
        pattern = sys.argv[2] if len(sys.argv) > 2 else "*.VVP_40.txt"
    else:
        data_dir = DEFAULT_DATA_DIR
        pattern = DEFAULT_PATTERN

    # Load and process profiles
    data = load_profiles(data_dir, pattern)

    # Unpack into module namespace
    results = data["results"]
    vpr_results = data["vpr_results"]
    timestamps = data["timestamps"]
    averaged = data["averaged"]

    # Print summary
    print("\n" + "=" * 60)
    print("Temporal Averaging Results")
    print("=" * 60)
    print(f"\nProfiles loaded: {len(results)}")
    print(f"Usable for VPR: {len(vpr_results)}")

    if averaged is not None:
        print(f"\nAveraged ground reflectivity: {averaged.z_ground_dbz:.1f} dBZ")
        print(f"Individual values: {[f'{v.z_ground_dbz:.1f}' for v in vpr_results]}")

    print("\n" + "-" * 60)
    print("Available variables:")
    print("  results     - List of ProcessedProfile objects")
    print("  vpr_results - List of VPRCorrectionResult objects")
    print("  timestamps  - Timestamps of the profiles")
    print("  averaged    - Temporally averaged VPRCorrectionResult")
    print("  fig         - Correction comparison plot")
    print("  fig2        - Ground reflectivity comparison plot")
    print("-" * 60 + "\n")

    # Create plots
    if vpr_results:
        fig = plot_correction_comparison(vpr_results, averaged)
        fig2 = plot_z_ground_comparison(vpr_results, averaged)
