"""Regression tests for the legacy reference-file parsers."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from .legacy_parser import (
    LegacyProfileFooter,
    LegacyProfileHeader,
    parse_legacy_cor,
    parse_legacy_corave,
    parse_legacy_profile,
)

DATA_DIR = Path(__file__).parent / "data"
PROFILE_PATH = DATA_DIR / "202508241100_KAN.VVP_40.profile"
COR_PATH = DATA_DIR / "202508241100_KAN.VVP_40.cor"


def _parse_profile_text(
    text: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[xr.Dataset, LegacyProfileHeader, LegacyProfileFooter]:
    monkeypatch.setattr(Path, "read_text", lambda self: text)
    return parse_legacy_profile(PROFILE_PATH)


def test_profile_reference_fields() -> None:
    ds, header, footer = parse_legacy_profile(PROFILE_PATH)

    assert header.radar_name == "KANKAANPAA"
    assert header.timestamp == datetime(2025, 8, 24, 11)
    assert header.step == 200
    assert header.mds == -45
    np.testing.assert_array_equal(ds.height, np.arange(126, 10000, 200))
    np.testing.assert_array_equal(ds.clim_dbz.values[:3], [13.5, 10.0, 9.1])
    np.testing.assert_array_equal(ds.kor_profile.values[:3], [32.8, 31.8, 31.9])
    np.testing.assert_array_equal(ds.zcount.values[:3], [5000, 5000, 5000])
    assert "sample_count" not in ds
    assert "gradient" not in ds
    assert footer.layer_num == 1
    assert footer.bottom_height == 126
    assert footer.top_height == 6726
    assert footer.evaporation == 3.0
    assert footer.bb_upper_amplitude == footer.bb_lower_amplitude == 0.0
    assert footer.quality_weight == 0.66
    assert footer.freezing_level == 0.0


@pytest.mark.parametrize("candidates", ["", "BB ehdokkaat\n1 1526\n2 2526\n"])
@pytest.mark.parametrize("multilayer", [False, True])
def test_profile_sections(
    candidates: str, multilayer: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = PROFILE_PATH.read_text()
    expected_ds, expected_header, expected_footer = parse_legacy_profile(PROFILE_PATH)
    extra_layer = (
        "2 7526 8326 12.0 0 0 1 0 0 0 0 0 0 0 0 0.0 0.0 3.0 "
        "0.66 24 8 2025 11 0\n"
        if multilayer
        else ""
    )

    ds, header, footer = _parse_profile_text(
        reference.rstrip() + "\n" + extra_layer + candidates + "\n",
        monkeypatch,
    )

    xr.testing.assert_identical(ds, expected_ds)
    assert header == expected_header
    assert footer == expected_footer


def test_profile_amplitude_and_evaporation_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = PROFILE_PATH.read_text().splitlines()
    columns = lines[-1].split()
    columns[9:19] = [
        "1",
        "1526",
        "8",
        "1926",
        "1126",
        "1926",
        "6.5",
        "4.5",
        "3.0",
        "0.42",
    ]
    lines[-1] = " ".join(columns)
    _, _, footer = _parse_profile_text("\n".join(lines), monkeypatch)

    assert footer.bright_band == 1
    assert footer.bb_height == 1526
    assert footer.bb_amplitude == 8
    assert footer.freezing_level == 1926
    assert footer.bb_bottom == 1126
    assert footer.bb_top == 1926
    assert footer.bb_upper_amplitude == 6.5
    assert footer.bb_lower_amplitude == 4.5
    assert footer.evaporation == 3.0
    assert footer.quality_weight == 0.42
    assert (footer.day, footer.month, footer.year, footer.hour, footer.minute) == (
        24,
        8,
        2025,
        11,
        0,
    )


@pytest.mark.parametrize("suffix", ["", "BB ehdokkaat\n1 1526\n"])
def test_profile_missing_footer_is_rejected(
    suffix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = PROFILE_PATH.read_text().splitlines()
    with pytest.raises(ValueError, match="Missing layer footer"):
        _parse_profile_text("\n".join(lines[:-1]) + "\n" + suffix, monkeypatch)


def test_profile_missing_description_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = PROFILE_PATH.read_text().splitlines()
    with pytest.raises(ValueError, match="Missing layer footer description"):
        _parse_profile_text("\n".join(lines[:-2] + lines[-1:]), monkeypatch)


def test_profile_malformed_data_is_not_silently_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = PROFILE_PATH.read_text().splitlines()
    lines[1] = "126 33.1 20.6 32.8 32.8 13.5"
    with pytest.raises(ValueError, match="Expected 7 profile columns"):
        _parse_profile_text("\n".join(lines), monkeypatch)


def test_profile_reflectivity_counts_are_not_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = PROFILE_PATH.read_text().splitlines()
    columns = lines[1].split()
    columns[6] = "10000"
    lines[1] = " ".join(columns)
    ds, _, _ = _parse_profile_text("\n".join(lines), monkeypatch)

    assert ds.zcount.sel(height=126).item() == 10000


def test_cor_reference_products_are_distinct() -> None:
    ds, header = parse_legacy_cor(COR_PATH)

    assert header.quality_weight == 0.66
    assert header.freezing_level_m == 0
    assert header.elevation_angles == [0.7, 1.5, 3.0, 3.3]
    assert ds.sizes["correction_type"] == 6
    np.testing.assert_array_equal(
        ds.correction_klim_db.sel(range_km=1), [0.0, 2.3, -3.5, -3.5, -3.5, -3.5]
    )
    np.testing.assert_array_equal(
        ds.correction_rain_db.sel(range_km=1), [1.0, -0.2, 0.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_array_equal(
        ds.beam_height_m.sel(range_km=1), [325, 825, 12, 26, 52, 57]
    )


@pytest.mark.parametrize("freezing_height", [0, -174, 1826])
@pytest.mark.parametrize("averaged", [False, True])
def test_correction_freezing_height_is_not_a_boolean_or_asl(
    freezing_height: int, averaged: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if averaged:
        header = (
            f"KANKAANPAA VVP_40 200 -45 24 08 2025 11 00 {freezing_height} "
            "ELEVS 0.7\n"
        )
        text = header + "1 0.0 1.0 325 2.3 -0.2 825 -3.5 0.0 12 8 4\n"
    else:
        lines = COR_PATH.read_text().splitlines()
        columns = lines[0].split()
        columns[16] = str(freezing_height)
        text = " ".join(columns) + "\n" + lines[1]
    monkeypatch.setattr(Path, "read_text", lambda self: text)

    parser = parse_legacy_corave if averaged else parse_legacy_cor
    ds, parsed_header = parser(COR_PATH)

    assert parsed_header.freezing_level_m == freezing_height
    assert ds.attrs["freezing_level_m"] == freezing_height
    if averaged:
        assert ds.count_klim.item() == 8
        assert ds.count_rain.item() == 4
