"""Temporal averaging of VPR correction profiles.

This module implements time-weighted averaging of VPR corrections
across multiple observation times. Newer profiles receive higher
weight, enabling smoother temporal evolution while favoring recent
observations.

Based on pystycappi_ka.pl from the Koistinen & Pohjola algorithm.

Algorithm:
    Profiles are weighted by their temporal proximity to the most
    recent observation. The newest profile receives weight 1.0,
    and weights decay linearly with age to a minimum of MIN_AGE_WEIGHT
    for the oldest profile.

    For each profile with timestamp t:
        age_fraction = (t_newest - t) / (t_newest - t_oldest)
        weight = 1.0 - age_fraction * (1.0 - MIN_AGE_WEIGHT)

    The weighted average is:
        correction_avg = Σ(correction[j] * w[j]) / Σ(w[j])
"""

from datetime import datetime
from typing import Literal

import numpy as np
import xarray as xr

from .vpr_correction import VPRCorrectionResult

# Minimum weight for the oldest profile in temporal averaging
# Matches the spirit of legacy 1/n weighting for typical n≈5-6 profiles
MIN_AGE_WEIGHT = 0.2


def average_corrections(
    results: list[VPRCorrectionResult],
    *,
    timestamps: list[datetime] | None = None,
    weighting: Literal["elapsed", "legacy"] = "elapsed",
) -> VPRCorrectionResult:
    """Compute time-weighted average of VPR correction profiles.

    Newer profiles receive linearly increasing weight. Profiles are
    sorted by timestamp before averaging.

    Args:
        results: VPRCorrectionResult objects to average. Must have
            at least one result. All must have identical correction variables
            and coordinates (range_km, cappi_height, elevation), with finite
            values. Blended products are averaged after per-profile blending.
        timestamps: Observation time for each result. If None, timestamps
            are extracted from each result's corrections.attrs['timestamp'].
        weighting: "elapsed" retains elapsed-time weighting with an oldest
            weight of 0.2. "legacy" uses rank weights (j + 1) / n, matching
            pystycappi_ka.pl after sorting from oldest to newest.

    Returns:
        VPRCorrectionResult with time-averaged corrections. The returned
        result has usable=True if any input was usable. Quality weights and
        ground references are averaged with the same temporal weights.

    Example:
        >>> from vprc import process_vvp, average_corrections
        >>> profiles = [process_vvp(f).vpr_correction for f in files]
        >>> avg = average_corrections(profiles)
        >>> avg.corrections['cappi_correction_db'].sel(range_km=100)

    Note:
        The averaging is performed in dB space (consistent with Perl
        implementation) rather than linear Z space.
    """
    if not results:
        raise ValueError("results must not be empty")
    if weighting not in ("elapsed", "legacy"):
        raise ValueError("weighting must be 'elapsed' or 'legacy'")

    # Extract timestamps if not provided
    if timestamps is None:
        timestamps = _extract_timestamps(results)

    if len(timestamps) != len(results):
        raise ValueError(
            f"timestamps length ({len(timestamps)}) must match "
            f"results length ({len(results)})"
        )

    # Sort by timestamp (oldest first)
    sorted_pairs = sorted(zip(timestamps, results, strict=True), key=lambda x: x[0])
    sorted_results = [r for _, r in sorted_pairs]
    n = len(sorted_results)

    reference = sorted_results[0].corrections
    for i, result in enumerate(sorted_results):
        ds = result.corrections
        if set(ds.data_vars) != set(reference.data_vars):
            raise ValueError(f"Result {i} has incompatible correction variables")
        if set(ds.coords) != set(reference.coords):
            raise ValueError(f"Result {i} has incompatible coordinates")
        for name in reference.data_vars:
            if ds[name].dims != reference[name].dims:
                raise ValueError(f"Result {i} has incompatible dimensions for {name}")
            if not np.isfinite(ds[name].values).all():
                raise ValueError(f"Result {i} contains non-finite values in {name}")
        if not np.isfinite(result.quality_weight) or result.quality_weight < 0:
            raise ValueError(f"Result {i} has invalid quality_weight")
        observed_quality = ds.attrs.get("quality_weight", result.quality_weight)
        if not np.isfinite(observed_quality) or observed_quality < 0:
            raise ValueError(f"Result {i} has invalid observed quality_weight")
        if not np.isfinite(result.z_ground_dbz):
            raise ValueError(f"Result {i} has invalid z_ground_dbz")
        if (result.z_ground_clim_dbz is None) != (
            sorted_results[0].z_ground_clim_dbz is None
        ):
            raise ValueError("Results must have compatible climatological references")
        if result.z_ground_clim_dbz is not None and not np.isfinite(
            result.z_ground_clim_dbz
        ):
            raise ValueError(f"Result {i} has invalid z_ground_clim_dbz")

    # Reject coordinate unions rather than diluting missing ranges with zeros.
    aligned = xr.align(*(r.corrections for r in sorted_results), join="exact")

    # Single profile: preserve values, but respect an explicit timestamp.
    if n == 1:
        result = sorted_results[0]
        ds = result.corrections.copy()
        ds.attrs["timestamp"] = sorted_pairs[0][0].isoformat()
        return VPRCorrectionResult(
            corrections=ds,
            z_ground_dbz=result.z_ground_dbz,
            z_ground_clim_dbz=result.z_ground_clim_dbz,
            quality_weight=result.quality_weight,
            usable=result.usable,
        )

    # Compute time-delta-based weights
    # Newest profile gets weight 1.0, oldest gets MIN_AGE_WEIGHT
    # Linear interpolation based on actual temporal position
    sorted_timestamps = [t for t, _ in sorted_pairs]
    weights = (
        _compute_age_weights(sorted_timestamps)
        if weighting == "elapsed"
        else np.arange(1, n + 1, dtype=float) / n
    )
    weight_sum = weights.sum()

    # Stack all correction datasets and compute weighted average
    stacked = xr.concat(
        aligned,
        dim="time",
        join="exact",
    )

    # Apply weights along time dimension
    weights_da = xr.DataArray(weights, dims=["time"])

    avg_ds = stacked.weighted(weights_da).mean(dim="time", skipna=False)

    # Weighted average of z_ground_dbz
    z_grounds = np.array([r.z_ground_dbz for r in sorted_results])
    avg_z_ground = float((z_grounds * weights).sum() / weight_sum)
    avg_quality = float(
        np.average([r.quality_weight for r in sorted_results], weights=weights)
    )
    avg_z_ground_clim = None
    if sorted_results[0].z_ground_clim_dbz is not None:
        clim_references = [
            r.z_ground_clim_dbz
            for r in sorted_results
            if r.z_ground_clim_dbz is not None
        ]
        avg_z_ground_clim = float(np.average(clim_references, weights=weights))

    # Copy attrs from newest profile, update timestamp to latest
    avg_ds.attrs = dict(sorted_results[-1].corrections.attrs)
    avg_ds.attrs["timestamp"] = sorted_pairs[-1][0].isoformat()
    avg_ds.attrs["z_ground_dbz"] = avg_z_ground
    avg_ds.attrs["z_ground_clim_dbz"] = avg_z_ground_clim
    # Dataset quality describes observations; result quality includes fallback.
    avg_ds.attrs["quality_weight"] = float(
        np.average(
            [
                r.corrections.attrs.get("quality_weight", r.quality_weight)
                for r in sorted_results
            ],
            weights=weights,
        )
    )
    avg_ds.attrs["climatology_only"] = all(
        r.corrections.attrs.get("climatology_only", False) for r in sorted_results
    )
    avg_ds.attrs["averaged_from_n"] = n
    avg_ds.attrs["temporal_weighting"] = weighting

    # Result is usable if any input was usable
    any_usable = any(r.usable for r in sorted_results)

    return VPRCorrectionResult(
        corrections=avg_ds,
        z_ground_dbz=avg_z_ground,
        z_ground_clim_dbz=avg_z_ground_clim,
        quality_weight=avg_quality,
        usable=any_usable,
    )


def _extract_timestamps(results: list[VPRCorrectionResult]) -> list[datetime]:
    """Extract timestamps from VPRCorrectionResult objects.

    Args:
        results: List of VPR correction results

    Returns:
        List of datetime objects extracted from each result's attrs

    Raises:
        ValueError: If any result is missing a valid timestamp
    """
    timestamps = []
    for i, r in enumerate(results):
        ts_str = r.corrections.attrs.get("timestamp")
        if not ts_str:
            raise ValueError(f"Result {i} is missing 'timestamp' attribute")
        try:
            ts = datetime.fromisoformat(ts_str)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Result {i} has invalid timestamp '{ts_str}': {e}") from e
        timestamps.append(ts)
    return timestamps


def _compute_age_weights(timestamps: list[datetime]) -> np.ndarray:
    """Compute time-delta-based weights for temporal averaging.

    The newest observation receives weight 1.0, the oldest receives
    MIN_AGE_WEIGHT, with linear interpolation for intermediate times.

    Args:
        timestamps: Sorted list of timestamps (oldest first)

    Returns:
        Array of weights, same length as timestamps
    """
    n = len(timestamps)
    if n == 1:
        return np.array([1.0])

    # Calculate time deltas from newest (last) timestamp
    newest = timestamps[-1]
    oldest = timestamps[0]
    time_span = (newest - oldest).total_seconds()

    if time_span == 0:
        # All timestamps identical: equal weights
        return np.ones(n)

    # age_fraction: 1.0 for oldest, 0.0 for newest
    # weight: MIN_AGE_WEIGHT for oldest, 1.0 for newest
    weights = np.array(
        [
            1.0 - ((newest - t).total_seconds() / time_span) * (1.0 - MIN_AGE_WEIGHT)
            for t in timestamps
        ]
    )

    return weights
