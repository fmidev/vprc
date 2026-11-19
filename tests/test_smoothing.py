"""Regressions for raw detection and sequential working-profile corrections."""

import numpy as np
import pytest
import xarray as xr

from vprc.clutter import remove_ground_clutter
from vprc.constants import MDS
from vprc.smoothing import (
    correct_lower_boundary_spike,
    correct_upper_boundary_spike,
    fill_gap,
    remove_isolated_echo,
    smooth_negative_spikes,
    smooth_positive_spikes,
    smooth_spikes,
)


def make_profile(values, *, zcount=None, raw=None):
    values = np.asarray(values, dtype=float)
    return xr.Dataset(
        {
            "lin_dbz": (
                "height",
                values.copy() if raw is None else np.asarray(raw, dtype=float),
            ),
            "corrected_dbz": ("height", values.copy()),
            "zcount": (
                "height",
                zcount if zcount is not None else [1000] * len(values),
            ),
            "sample_count": ("height", [0] * len(values)),
        },
        coords={"height": np.arange(100, 100 + 200 * len(values), 200)},
    )


@pytest.mark.parametrize("upper,expected", [(22, 18), (18, 20), (20, 20)])
def test_lower_boundary_requires_two_missing_levels(upper, expected):
    ds = make_profile(
        [MDS, MDS, MDS, 25, 20, upper, upper],
        zcount=[0, 0, 0, 1000, 1000, 1000, 1000],
    )
    correct_lower_boundary_spike(ds, 500)
    assert float(ds.corrected_dbz.sel(height=700)) == expected


def test_lower_boundary_does_not_correct_second_valid_level():
    ds = make_profile([MDS, MDS, 15, 25, 20, 22], zcount=[0, 0, 1000, 1000, 1000, 1000])
    correct_lower_boundary_spike(ds, 500)
    assert float(ds.corrected_dbz.sel(height=700)) == 25


@pytest.mark.parametrize("lower,expected", [(22, 18), (18, 20), (20, 20)])
def test_upper_boundary_extrapolates_only_negative_gradient(lower, expected):
    ds = make_profile([15, lower, 20, 25, MDS, MDS], zcount=[1000] * 4 + [0, 0])
    correct_upper_boundary_spike(ds, 500)
    assert float(ds.corrected_dbz.sel(height=700)) == expected


@pytest.mark.parametrize(
    "function,values,expected",
    [
        (smooth_positive_spikes, [16, 20, 16], [16, 52 / 3, 16]),
        (smooth_positive_spikes, [15, 30, 15], [15, 15, 15]),
        (smooth_positive_spikes, [15, 18, 15], [15, 18, 15]),
        (smooth_positive_spikes, [15, 22, 20], [15, 22, 20]),
        (smooth_negative_spikes, [20, 16, 20], [20, 56 / 3, 20]),
        (smooth_negative_spikes, [25, 5, 25], [25, 25, 25]),
        (smooth_negative_spikes, [20, 17, 20], [20, 17, 20]),
        (smooth_negative_spikes, [25, 18, 20], [25, 18, 20]),
        (smooth_positive_spikes, [15, 20, 25], [15, 20, 25]),
        (smooth_negative_spikes, [15, 20, 25], [15, 20, 25]),
    ],
)
def test_spike_thresholds_and_exact_averages(function, values, expected):
    ds = make_profile(values)
    original = ds.copy(deep=True)
    np.testing.assert_allclose(function(ds).corrected_dbz, expected)
    xr.testing.assert_identical(ds, original)


@pytest.mark.parametrize("function", [smooth_positive_spikes, smooth_negative_spikes])
@pytest.mark.parametrize("count", [0, 30])
def test_spikes_require_strict_zcount_threshold(function, count):
    values = [20, 35, 20] if function is smooth_positive_spikes else [20, 5, 20]
    ds = make_profile(values, zcount=[1000, count, 1000])
    ds["sample_count"][:] = 1000
    np.testing.assert_allclose(function(ds).corrected_dbz, values)


def test_positive_detection_uses_raw_but_averages_working_values():
    ds = make_profile([16, 17, 16], raw=[16, 30, 16])
    np.testing.assert_allclose(smooth_positive_spikes(ds).corrected_dbz, [16, 16, 16])


def test_negative_detection_uses_raw_but_averages_working_values():
    ds = make_profile([20, 19, 20], raw=[20, 0, 20])
    np.testing.assert_allclose(smooth_negative_spikes(ds).corrected_dbz, [20, 20, 20])


@pytest.mark.parametrize(
    "counts,values,expected",
    [
        ([1000, 1000, 0, 1000, 0], [10, 15, MDS, 25, MDS], 20),
        ([0, 1000, 0, 1000, 1000], [MDS, 12, MDS, 18, 20], 15),
        ([0, 1000, 0, 1000, 0], [MDS, 12, MDS, 18, MDS], MDS),
        ([1000, 1000, 0, 1000, 0], [10, 15, MDS, MDS, MDS], MDS),
        ([1000] * 5, [10, 15, 12, 25, 30], 12),
        ([1000, 1000, 30, 1000, 1000], [10, 15, MDS, 25, 30], MDS),
    ],
)
def test_gap_patterns(counts, values, expected):
    ds = make_profile(values, zcount=counts)
    fill_gap(ds, 500)
    assert float(ds.corrected_dbz.sel(height=500)) == expected


def test_gap_can_be_filled_at_second_level_with_upper_support():
    ds = make_profile([10, MDS, 20, 20], zcount=[1000, 0, 1000, 1000])
    fill_gap(ds, 300)
    assert float(ds.corrected_dbz.sel(height=300)) == 15


@pytest.mark.parametrize("support", [None, 0, 1, 3, 4])
def test_isolated_echo_removal_and_supported_echo(support):
    counts = [0, 0, 1000, 0, 0]
    if support is not None:
        counts[support] = 1000
    ds = make_profile([MDS, MDS, 25, MDS, MDS], zcount=counts)
    remove_isolated_echo(ds, 500)
    assert float(ds.corrected_dbz.sel(height=500)) == (MDS if support is None else 25)


def test_isolated_echo_at_profile_edge():
    ds = make_profile([25, MDS, MDS], zcount=[1000, 0, 0])
    remove_isolated_echo(ds, 100)
    assert float(ds.corrected_dbz[0]) == MDS


@pytest.mark.parametrize(
    "values,counts,expected",
    [
        ([0, 10, 0, 10, 0, 0, 0, 0], [1000] * 8, [0, 0, 10 / 3, 5 / 3, 0, 0, 0, 0]),
        (
            [MDS, MDS, 10, 20, MDS, MDS],
            [0, 0, 1000, 1000, 0, 0],
            [MDS, MDS, 15, 15, MDS, MDS],
        ),
        (
            [20, 20, 35, 35, 20, 20, 20, 20],
            [1000, 1000, 0, 0, 1000, 1000, 1000, 1000],
            [20, 20, MDS, MDS, 20, 20, 20, 20],
        ),
        (
            [MDS, MDS, MDS, 25, 20, 22, 22, 22],
            [0, 0, 0, 1000, 1000, 1000, 1000, 1000],
            [MDS, MDS, MDS, 18, 20, 22, 22, 22],
        ),
    ],
)
def test_full_smoothing_matches_perl_sequences(values, counts, expected):
    ds = make_profile(values, zcount=counts)
    original = ds.copy(deep=True)
    np.testing.assert_allclose(smooth_spikes(ds).corrected_dbz, expected)
    xr.testing.assert_identical(ds, original)


def test_smoothing_preserves_clutter_extrapolation_below_start():
    ds = make_profile([40, 35, 30, 20, 22, 24, 24])
    corrected = remove_ground_clutter(ds)
    result = smooth_spikes(corrected)
    np.testing.assert_allclose(result.corrected_dbz[:3], [23, 22, 21])
    assert result.attrs["smoothing_start_height_m"] == 700


def test_time_profiles_are_processed_independently():
    first = make_profile([0, 10, 0, 10, 0, 0, 0, 0])
    second = make_profile([20] * 8)
    ds = xr.concat([first, second], dim=xr.IndexVariable("time", [0, 1]))
    actual = smooth_spikes(ds)
    for t in ds.time.values:
        xr.testing.assert_allclose(
            actual.sel(time=t, drop=True), smooth_spikes(ds.sel(time=t, drop=True))
        )


def test_empty_profile():
    ds = make_profile([])
    xr.testing.assert_identical(smooth_spikes(ds), ds)


def test_integer_input_does_not_truncate_fractional_corrections():
    ds = make_profile([20, 35, 25])
    ds["corrected_dbz"] = ds.corrected_dbz.astype(int)
    assert float(smooth_positive_spikes(ds).corrected_dbz[1]) == 22.5


def test_levels_above_profile_top_are_only_neighbours():
    """allprof_prodx2.pl analyses levels below 10 km and reads higher ones.

    Echo continuing above the top keeps 9900 m from being smoothed as a spike;
    the higher levels are then dropped, including their missing-level entries.
    """
    ds = make_profile([20.0] * 49 + [30.0, 30.0, 30.0])  # 100 ... 10300 m
    ds.attrs["missing_levels_m"] = [300, 10100]
    result = smooth_spikes(ds)
    assert int(result.height.max()) == 9900
    assert float(result.corrected_dbz.sel(height=9900)) == 30
    assert result.attrs["missing_levels_m"] == [300]


def test_filled_absent_level_takes_the_filled_raw_value():
    """An absent level's MDS placeholder must not stay in raw-value arithmetic
    once smoothing has filled it; an unfilled absent level stays no echo."""
    values = [10, 15, MDS, 25, 25, MDS, MDS, MDS]
    counts = [1000, 1000, 0, 1000, 1000, 0, 0, 0]
    ds = make_profile(values, zcount=counts)
    ds.attrs["missing_levels_m"] = [500, 1100]
    result = smooth_spikes(ds)
    assert float(result.corrected_dbz.sel(height=500)) == 20
    assert float(result.lin_dbz.sel(height=500)) == 20
    assert float(result.lin_dbz.sel(height=1100)) == MDS
    # A present level with few samples keeps its measured raw value.
    ds.attrs["missing_levels_m"] = []
    assert float(smooth_spikes(ds).lin_dbz.sel(height=500)) == MDS
