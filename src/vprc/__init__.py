# SPDX-FileCopyrightText: 2025-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT
"""VPR correction for weather radar data.

This package implements the Koistinen & Pohjola VPR (Vertical Profile
Reflectivity) correction algorithm for weather radar data.

Main entry point:
    process_vvp(path, **radar_metadata) -> ProcessedProfile

Individual processing steps:
    io.read_vvp() - Parse VVP profile files
    clutter.remove_ground_clutter() - Remove ground clutter
    smoothing.apply_spike_smoothing() - Smooth profile spikes
    classification.classify_profile() - Segment and classify layers
    bright_band.detect_bright_band() - Detect melting layer
"""

from dataclasses import dataclass
from pathlib import Path

import xarray as xr

from .io import read_vvp
from .clutter import remove_ground_clutter
from .smoothing import smooth_spikes
from .classification import classify_profile, ProfileClassification, LayerType
from .bright_band import detect_bright_band, BrightBandResult


@dataclass
class ProcessedProfile:
    """Result of processing a VVP profile through the full pipeline.

    Attributes:
        dataset: xarray Dataset with corrected_dbz after all processing
        classification: Layer classification results
        bright_band: Bright band detection results
        usable_for_vpr: True if profile is suitable for VPR correction
    """

    dataset: xr.Dataset
    classification: ProfileClassification
    bright_band: BrightBandResult

    @property
    def usable_for_vpr(self) -> bool:
        """Check if the profile is usable for VPR correction."""
        return self.classification.usable_for_vpr


def process_vvp(
    path: str | Path,
    *,
    antenna_height_m: int | None = None,
    lowest_level_offset_m: int | None = None,
    freezing_level_m: float | None = None,
    **kwargs,
) -> ProcessedProfile:
    """Process a VVP profile through the full correction pipeline.

    This is the main entry point for VPR correction. It reads a VVP file,
    applies ground clutter removal, spike smoothing, layer classification,
    and bright band detection.

    Args:
        path: Path to the VVP profile file
        antenna_height_m: Antenna height above sea level (m)
        lowest_level_offset_m: Offset from antenna to lowest profile level (m)
        freezing_level_m: Freezing level from NWP (m above sea level)
        **kwargs: Additional metadata passed to read_vvp()

    Returns:
        ProcessedProfile with corrected dataset, classification, and BB info

    Example:
        >>> result = process_vvp(
        ...     "profile.txt",
        ...     freezing_level_m=2500,
        ... )
        >>> if result.usable_for_vpr:
        ...     print(f"Profile type: {result.classification.profile_type}")
        ...     if result.bright_band.detected:
        ...         print(f"BB at {result.bright_band.peak_height}m")

    Note:
        For Airflow integration, this function is designed to be called
        from a @task.docker decorator with radar_config passed as kwargs.
    """
    # Build metadata dict from explicit args
    radar_metadata = {}
    if antenna_height_m is not None:
        radar_metadata["antenna_height_m"] = antenna_height_m
    if lowest_level_offset_m is not None:
        radar_metadata["lowest_level_offset_m"] = lowest_level_offset_m
    if freezing_level_m is not None:
        radar_metadata["freezing_level_m"] = freezing_level_m
    radar_metadata.update(kwargs)

    # Step 1: Read VVP file
    ds = read_vvp(path, radar_metadata=radar_metadata if radar_metadata else None)

    # Step 2: Ground clutter removal
    ds = remove_ground_clutter(ds)

    # Step 3: Spike smoothing
    ds = smooth_spikes(ds)

    # Step 4: Layer classification
    classification = classify_profile(ds)

    # Step 5: Bright band detection
    # Use lowest layer top as search limit if available
    layer_top = None
    if classification.lowest_layer is not None:
        layer_top = classification.lowest_layer.top_height

    bright_band = detect_bright_band(ds, layer_top=layer_top)

    return ProcessedProfile(
        dataset=ds,
        classification=classification,
        bright_band=bright_band,
    )


__all__ = [
    "process_vvp",
    "ProcessedProfile",
    "ProfileClassification",
    "LayerType",
    "BrightBandResult",
    "read_vvp",
]
