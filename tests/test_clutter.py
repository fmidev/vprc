"""Numerical regressions for the ground-clutter stage."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from vprc import process_vvp
from vprc.clutter import compute_gradient, remove_ground_clutter
from vprc.config import get_radar_metadata
from vprc.constants import MDS
from vprc.io import read_vvp

DATA_DIR = Path(__file__).parent / "data"
GAPPED_VVP_FILE = DATA_DIR / "202611112000_KES.VVP_40.txt"


def make_profile(values, *, zcount=None, count=None, freezing_level=2000):
    values = np.asarray(values, dtype=float)
    return xr.Dataset(
        {
            "lin_dbz": ("height", values.copy()),
            "corrected_dbz": ("height", values.copy()),
            "zcount": (
                "height",
                zcount if zcount is not None else [1000] * len(values),
            ),
            "sample_count": (
                "height",
                count if count is not None else [0] * len(values),
            ),
        },
        coords={"height": np.arange(100, 100 + 200 * len(values), 200)},
        attrs={"freezing_level_m": freezing_level},
    )


def test_forward_gradient_uses_raw_reflectivity_and_zcount():
    ds = make_profile([10, 15, 12, 20], count=[0] * 4)
    ds["corrected_dbz"][:] = 20
    np.testing.assert_allclose(
        compute_gradient(ds), [0.025, -0.015, 0.04, np.nan], equal_nan=True
    )


@pytest.mark.parametrize("count", [0, 29, 30, np.nan])
def test_gradient_requires_more_than_thirty_reflectivity_samples(count):
    ds = make_profile(
        [10, 15, 20, 25], zcount=[1000, count, 1000, 1000], count=[1000] * 4
    )
    np.testing.assert_allclose(
        compute_gradient(ds), [np.nan, np.nan, 0.025, np.nan], equal_nan=True
    )


def test_custom_sample_threshold():
    ds = make_profile([10, 15, 20], zcount=[50] * 3)
    assert np.isnan(compute_gradient(ds, min_samples=50)).all()
    assert float(compute_gradient(ds, min_samples=49)[0]) == pytest.approx(0.025)


@pytest.mark.parametrize(
    "values,expected,start,clutter",
    [
        ([20, 22, 24, 26, 28], [20, 22, 24, 26, 28], 100, False),
        ([35, 20, 22, 24, 26], [21, 20, 22, 24, 26], 300, True),
        ([40, 35, 20, 22, 24], [22, 21, 20, 22, 24], 500, True),
        # Correct the Perl duplicated-h1 typo rather than preserving it.
        ([40, 35, 30, 20, 22], [23, 22, 21, 20, 22], 700, True),
        ([18, 35, 20, 22, 24], [18, 21, 20, 22, 24], 500, True),
        ([MDS, 20, 22, 24, 26], [MDS, 20, 22, 24, 26], 100, False),
    ],
)
def test_exact_clutter_corrections(values, expected, start, clutter):
    ds = make_profile(values)
    before = ds.copy(deep=True)
    result = remove_ground_clutter(ds)
    np.testing.assert_allclose(result.corrected_dbz, expected)
    assert result.attrs["smoothing_start_height_m"] == start
    assert result.attrs["clutter_detected"] is clutter
    xr.testing.assert_identical(ds, before)


@pytest.mark.parametrize("freezing_level", [0.1, 500, 1000])
def test_skip_clutter_up_to_and_including_one_kilometre(freezing_level):
    ds = make_profile([35, 20, 22, 24], freezing_level=freezing_level)
    result = remove_ground_clutter(ds)
    np.testing.assert_equal(result.corrected_dbz.values, ds.corrected_dbz.values)
    assert result.attrs["clutter_detected"] is False


@pytest.mark.parametrize("freezing_level", [None, -200, 0, 1000.1, 2000])
def test_clutter_outside_protected_freezing_interval(freezing_level):
    ds = make_profile([35, 20, 22, 24], freezing_level=freezing_level)
    assert float(remove_ground_clutter(ds).corrected_dbz[0]) == 21


def test_echo_confined_below_level_four_is_removed_entirely():
    """Perl: "if there is no echo in levels 1-4, remove the clutter entirely".

    As run, Perl checked levels 2-5 and kept the lowest level's clutter
    extrapolation (20 + 2 dB) as an isolated echo.
    """
    ds = make_profile([40, 35, 20, MDS, MDS], zcount=[1000, 1000, 1000, 0, 0])
    result = remove_ground_clutter(ds)
    np.testing.assert_equal(result.corrected_dbz.values, [MDS] * 5)
    assert result.attrs["clutter_detected"] is False


@pytest.mark.parametrize(
    "zcount,expected",
    [
        # Single gaps between reliable levels: clear only the gap, which spike
        # smoothing fills, rather than detach the layer from the ground.
        ([1000, 20, 1000, 1000, 1000, 1000], [30, MDS, 32, 33, 34, 35]),
        ([1000, 1000, 20, 1000, 1000, 1000], [30, 31, MDS, 33, 34, 35]),
        ([1000, 1000, 1000, 20, 1000, 1000], [30, 31, 32, MDS, 34, 35]),
        # A gap at level 5 is outside levels 1-4. Perl as run cleared the
        # reliable levels 2-4 below it.
        ([1000, 1000, 1000, 1000, 0, 1000], [30, 31, 32, 33, 34, 35]),
        # Unreliable lowest level, or a gap of two levels: clear up to it.
        ([20, 1000, 1000, 1000, 1000, 1000], [MDS, 31, 32, 33, 34, 35]),
        ([1000, 20, 20, 1000, 1000, 1000], [MDS, MDS, MDS, 33, 34, 35]),
    ],
)
def test_low_zcount_cleanup_near_the_ground(zcount, expected):
    ds = make_profile([30, 31, 32, 33, 34, 35], zcount=zcount)
    result = remove_ground_clutter(ds)
    np.testing.assert_equal(result.corrected_dbz.values, expected)


def test_empty_profile_is_supported():
    ds = make_profile([])
    xr.testing.assert_identical(remove_ground_clutter(ds), ds)


def test_level_missing_from_vvp_file_does_not_break_clutter_removal():
    """Regression: an absent 500 m ASL KES level raised ``KeyError: 326``.

    The 526 m level triggers the three-level correction, which also writes
    126 and 326 m. The absent 326 m level is a single gap between reliable
    levels, so only it is cleared and the clutter correction stands. Perl
    also cleared the gap but reset the clutter flag.
    """
    ds = read_vvp(GAPPED_VVP_FILE, {"freezing_level_m": 126})
    result = remove_ground_clutter(ds)
    np.testing.assert_allclose(result.corrected_dbz[:4], [-21.6, MDS, -23.6, -24.6])
    assert result.attrs["smoothing_start_height_m"] == 726
    assert result.attrs["clutter_detected"] is True


def test_absent_fifth_level_keeps_reliable_levels_below():
    """Kuopio 2025-12-15 02 UTC lacks only 832 m above the antenna.

    Perl as run cleared levels 2-4 (ZCnt 372-4984) below that gap, and its
    snow clutter step turned the remaining lowest level into a -42 dBZ
    one-level layer. The echo reaches from the ground to 1432 m.
    """
    path = DATA_DIR / "202512150200_KUO.VVP_40.txt"
    metadata = get_radar_metadata("KUO")
    assert metadata is not None
    freezing_level = metadata["antenna_height_m"]
    ds = read_vvp(path, {"freezing_level_m": freezing_level})
    assert ds.attrs["missing_levels_m"] == [832]
    result = remove_ground_clutter(ds)
    assert (result.corrected_dbz.sel(height=[32, 232, 432, 632]) > MDS).all()
    layer = process_vvp(
        path, freezing_level_m=freezing_level
    ).classification.lowest_layer
    assert layer is not None
    assert (layer.bottom_height, layer.top_height) == (32, 1432)


def test_non_contiguous_heights_raise_value_error():
    ds = make_profile([30, MDS, 35, 20]).drop_sel(height=300)
    with pytest.raises(ValueError, match="contiguous 200 m"):
        remove_ground_clutter(ds)
