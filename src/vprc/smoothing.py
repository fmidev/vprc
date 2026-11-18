"""Sequential spike and echo-boundary processing from allprof_prodx2.pl."""

import numpy as np
import xarray as xr

from .constants import (
    LARGE_NEGATIVE_SPIKE_THRESHOLD,
    LARGE_POSITIVE_SPIKE_THRESHOLD,
    MDS,
    MIN_SAMPLES,
    PROFILE_TOP_M,
    SPIKE_AMPLITUDE_THRESHOLD,
    STEP,
)


def _at(ds: xr.Dataset, variable: str, height: int, missing: float) -> xr.DataArray:
    """Return a height slice, treating absent boundary levels as missing."""
    if height in ds.height.values:
        return ds[variable].sel(height=height, drop=True)
    return xr.full_like(ds[variable].isel(height=0, drop=True), missing, dtype=float)


def _set(
    ds: xr.Dataset,
    height: int,
    condition: xr.DataArray,
    value: xr.DataArray | float,
) -> None:
    if height in ds.height.values:
        if not np.issubdtype(ds["corrected_dbz"].dtype, np.floating):
            ds["corrected_dbz"] = ds["corrected_dbz"].astype(float)
        original = ds["corrected_dbz"].sel(height=height, drop=True)
        ds["corrected_dbz"].loc[{"height": height}] = xr.where(condition, value, original)


def correct_lower_boundary_spike(
    ds: xr.Dataset, height: int, sample_threshold: int = MIN_SAMPLES
) -> xr.Dataset:
    """Correct the first echo above two missing levels, in place."""
    count = lambda h: _at(ds, "zcount", h, 0)
    corrected = lambda h: _at(ds, "corrected_dbz", h, MDS)
    raw = _at(ds, "lin_dbz", height + STEP, MDS)
    above = corrected(height + 2 * STEP)
    pattern = (
        (count(height - STEP) < sample_threshold)
        & (count(height) < sample_threshold)
        & (count(height + STEP) > sample_threshold)
        & (count(height + 2 * STEP) > sample_threshold)
        & (raw > above)
    )
    slope = corrected(height + 3 * STEP) - above
    _set(ds, height + STEP, pattern, above - xr.where(slope > 0, slope, 0))
    return ds


def correct_upper_boundary_spike(
    ds: xr.Dataset, height: int, sample_threshold: int = MIN_SAMPLES
) -> xr.Dataset:
    """Correct the last echo below two missing levels, in place."""
    count = lambda h: _at(ds, "zcount", h, 0)
    corrected = lambda h: _at(ds, "corrected_dbz", h, MDS)
    current = corrected(height)
    pattern = (
        (count(height) > sample_threshold)
        & (count(height + STEP) > sample_threshold)
        & (count(height + 2 * STEP) < sample_threshold)
        & (count(height + 3 * STEP) < sample_threshold)
        & (_at(ds, "lin_dbz", height + STEP, MDS) > current)
    )
    slope = current - corrected(height - STEP)
    _set(ds, height + STEP, pattern, current + xr.where(slope < 0, slope, 0))
    return ds


def _smooth_at(
    ds: xr.Dataset,
    height: int,
    positive: bool,
    sample_threshold: int,
    spike_threshold: float,
    large_spike_threshold: float,
) -> None:
    """Detect against raw values, average the current working profile."""
    levels = (height, height + STEP, height + 2 * STEP)
    raw = [_at(ds, "lin_dbz", h, MDS) for h in levels]
    working = [_at(ds, "corrected_dbz", h, MDS) for h in levels]
    valid = (
        (_at(ds, "zcount", levels[0], 0) > sample_threshold)
        & (_at(ds, "zcount", levels[1], 0) > sample_threshold)
        & (_at(ds, "zcount", levels[2], 0) > sample_threshold)
    )
    direction = 1 if positive else -1
    spike = valid & ((raw[1] - raw[0]) * direction > spike_threshold) & (
        (raw[1] - raw[2]) * direction > spike_threshold
    )
    average = (working[0] + working[1] + working[2]) / 3
    residual = raw[1] - average
    large = (
        residual > large_spike_threshold
        if positive else residual < large_spike_threshold
    )
    correction = xr.where(large, (working[0] + working[2]) / 2, average)
    _set(ds, height + STEP, spike, correction)


def _smooth_spikes_rolling(
    ds: xr.Dataset,
    spike_type: str,
    sample_threshold: int = MIN_SAMPLES,
    spike_threshold: float = SPIKE_AMPLITUDE_THRESHOLD,
    large_spike_threshold: float | None = None,
) -> xr.Dataset:
    """Compatibility helper; sequential updates are required for numerical parity."""
    if spike_type not in ("positive", "negative"):
        raise ValueError(f"Unknown spike type: {spike_type}")
    positive = spike_type == "positive"
    if large_spike_threshold is None:
        large_spike_threshold = (
            LARGE_POSITIVE_SPIKE_THRESHOLD
            if positive else LARGE_NEGATIVE_SPIKE_THRESHOLD
        )
    result = ds.copy(deep=True)
    for height in result.height.values:
        _smooth_at(
            result, int(height), positive, sample_threshold,
            spike_threshold, large_spike_threshold,
        )
    return result


def smooth_positive_spikes(
    ds: xr.Dataset,
    sample_threshold: int = MIN_SAMPLES,
    spike_threshold: float = SPIKE_AMPLITUDE_THRESHOLD,
    large_spike_threshold: float = LARGE_POSITIVE_SPIKE_THRESHOLD,
) -> xr.Dataset:
    """Smooth raw positive spikes using evolving working values."""
    return _smooth_spikes_rolling(
        ds, "positive", sample_threshold, spike_threshold, large_spike_threshold
    )


def smooth_negative_spikes(
    ds: xr.Dataset,
    sample_threshold: int = MIN_SAMPLES,
    spike_threshold: float = SPIKE_AMPLITUDE_THRESHOLD,
    large_spike_threshold: float = LARGE_NEGATIVE_SPIKE_THRESHOLD,
) -> xr.Dataset:
    """Smooth raw negative spikes using evolving working values."""
    return _smooth_spikes_rolling(
        ds, "negative", sample_threshold, spike_threshold, large_spike_threshold
    )


def fill_gap(
    ds: xr.Dataset, height: int, sample_threshold: int = MIN_SAMPLES, mds: float = MDS
) -> xr.Dataset:
    """Fill a missing level with two supporting echoes on either side."""
    count = lambda h: _at(ds, "zcount", h, 0)
    corrected = lambda h: _at(ds, "corrected_dbz", h, mds)
    valid = lambda h: (count(h) > sample_threshold) & (corrected(h) > mds)
    pattern = (
        (count(height) < sample_threshold)
        & (corrected(height) >= mds)
        & valid(height - STEP)
        & valid(height + STEP)
        & (valid(height - 2 * STEP) | valid(height + 2 * STEP))
    )
    _set(ds, height, pattern, (corrected(height - STEP) + corrected(height + STEP)) / 2)
    return ds


def remove_isolated_echo(
    ds: xr.Dataset, height: int, sample_threshold: int = MIN_SAMPLES, mds: float = MDS
) -> xr.Dataset:
    """Remove a single echo with two missing neighbours on both sides."""
    count = lambda h: _at(ds, "zcount", h, 0)
    pattern = count(height) > sample_threshold
    for offset in (-2 * STEP, -STEP, STEP, 2 * STEP):
        pattern = pattern & (count(height + offset) < sample_threshold)
    _set(ds, height, pattern, mds)
    return ds


def _smooth_two_point_layer(ds: xr.Dataset, height: int) -> None:
    if height <= 400:
        return
    count = lambda h: _at(ds, "zcount", h, 0)
    pattern = (count(height) > MIN_SAMPLES) & (count(height + STEP) > MIN_SAMPLES)
    for offset in (-2 * STEP, -STEP, 2 * STEP, 3 * STEP):
        pattern = pattern & (count(height + offset) < MIN_SAMPLES)
    average = (
        _at(ds, "lin_dbz", height, MDS) + _at(ds, "lin_dbz", height + STEP, MDS)
    ) / 2
    _set(ds, height, pattern, average)
    _set(ds, height + STEP, pattern, average)


def smooth_spikes(ds: xr.Dataset) -> xr.Dataset:
    """Apply all operations in height order, preserving clutter extrapolation.

    Operations must be interleaved: later averages use earlier corrections,
    while spike detection always uses the original ``lin_dbz``.

    Levels at or above ``PROFILE_TOP_M`` are used only as neighbours and are
    dropped from the result, as in allprof_prodx2.pl. Each operation writes
    only its own level or the one above, so they cannot affect lower levels.
    """
    result = ds.copy(deep=True)
    if not result.sizes["height"]:
        return result
    start = result.attrs.get("smoothing_start_height_m", int(result.height.values[0]))
    for height in result.height.values:
        height = int(height)
        if height < start:
            continue
        correct_lower_boundary_spike(result, height)
        correct_upper_boundary_spike(result, height)
        _smooth_at(
            result, height, True, MIN_SAMPLES,
            SPIKE_AMPLITUDE_THRESHOLD, LARGE_POSITIVE_SPIKE_THRESHOLD,
        )
        _smooth_at(
            result, height, False, MIN_SAMPLES,
            SPIKE_AMPLITUDE_THRESHOLD, LARGE_NEGATIVE_SPIKE_THRESHOLD,
        )
        _set(
            result, height,
            ~(_at(result, "zcount", height, 0) > MIN_SAMPLES), MDS,
        )
        fill_gap(result, height)
        remove_isolated_echo(result, height)
        _smooth_two_point_layer(result, height)
    result = result.sel(height=result.height < PROFILE_TOP_M)
    if "missing_levels_m" in result.attrs:
        result.attrs["missing_levels_m"] = [
            height
            for height in result.attrs["missing_levels_m"]
            if height < PROFILE_TOP_M
        ]
    return result
