"""Numerical regressions for the ground-clutter stage."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from vprc.clutter import compute_gradient, remove_ground_clutter
from vprc.constants import MDS
from vprc.io import read_vvp

GAPPED_VVP_FILE = Path(__file__).parent / "data" / "202611112000_KES.VVP_40.txt"


def make_profile(values, *, zcount=None, count=None, freezing_level=2000):
    values = np.asarray(values, dtype=float)
    return xr.Dataset(
        {
            "lin_dbz": ("height", values.copy()),
            "corrected_dbz": ("height", values.copy()),
            "zcount": ("height", zcount if zcount is not None else [1000] * len(values)),
            "sample_count": ("height", count if count is not None else [0] * len(values)),
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
    ds = make_profile([10, 15, 20, 25], zcount=[1000, count, 1000, 1000], count=[1000] * 4)
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


def test_missing_upper_gradient_removes_lower_echo_and_clears_clutter():
    ds = make_profile([40, 35, 20, MDS, MDS], zcount=[1000, 1000, 1000, 0, 0])
    result = remove_ground_clutter(ds)
    np.testing.assert_equal(result.corrected_dbz.values, [MDS] * 5)
    assert result.attrs["clutter_detected"] is False


def test_empty_profile_is_supported():
    ds = make_profile([])
    xr.testing.assert_identical(remove_ground_clutter(ds), ds)


def test_level_missing_from_vvp_file_does_not_break_clutter_removal():
    """Regression: an absent 500 m ASL KES level raised ``KeyError: 326``.

    The 526 m level triggers the three-level correction, which also writes
    326 m. As in allprof_prodx2.pl, the absent level has no gradient, so the
    levels up to it are removed and the clutter flag is cleared.
    """
    ds = read_vvp(GAPPED_VVP_FILE, {"freezing_level_m": 126})
    result = remove_ground_clutter(ds)
    np.testing.assert_allclose(result.corrected_dbz[:4], [MDS, MDS, -23.6, -24.6])
    assert result.attrs["smoothing_start_height_m"] == 726
    assert result.attrs["clutter_detected"] is False


def test_non_contiguous_heights_raise_value_error():
    ds = make_profile([30, MDS, 35, 20]).drop_sel(height=300)
    with pytest.raises(ValueError, match="contiguous 200 m"):
        remove_ground_clutter(ds)
