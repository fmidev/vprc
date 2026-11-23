"""End-to-end tests comparing Python implementation against legacy Perl output.

These tests validate that the Python reimplementation produces results
matching the reference Perl implementation (allprof_prodx2.pl).

Test data: tests/data/*.VVP_40.* files (see tests/README.md)
- .txt: Input VVP profile
- .profile: Expected corrected profile output
- .cor: Expected correction output

Height coordinate system:
Both Python and legacy use heights above antenna (meters). The legacy code
converts sea-level heights from VVP files using:
    height_above_antenna = height_sea_level - antenna_height_m
Levels below antenna are dropped (negative heights).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vprc import (
    BBRejectionReason,
    PrecipitationType,
    get_radar_metadata,
    process_vvp,
)
from vprc.constants import MAX_CORRECTION_DB, MDS, MIN_SAMPLES

from .legacy_parser import parse_legacy_profile

# Paths to test data
DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "202508241100_KAN.VVP_40.txt"
EXPECTED_PROFILE = DATA_DIR / "202508241100_KAN.VVP_40.profile"


@pytest.fixture(scope="module")
def expected_profile():
    """Load the expected legacy profile output."""
    ds, header, footer = parse_legacy_profile(EXPECTED_PROFILE)
    return ds, header, footer


@pytest.fixture(scope="module")
def processed_result(expected_profile):
    """Run with the reference freezing height, converted to the API's ASL."""
    _, _, footer = expected_profile
    metadata = get_radar_metadata("KAN")
    assert metadata is not None
    antenna_height = metadata["antenna_height_m"]
    return process_vvp(
        INPUT_FILE,
        freezing_level_m=footer.freezing_level + antenna_height,
    )


@pytest.fixture(scope="module")
def legacy_cor():
    """Load the legacy .cor file."""
    from .legacy_parser import parse_legacy_cor

    cor_path = DATA_DIR / "202508241100_KAN.VVP_40.cor"
    return parse_legacy_cor(cor_path)


@pytest.fixture(scope="module")
def python_profile(processed_result, legacy_cor, expected_profile):
    """Use the reference's range and minimum elevation, not API defaults."""
    from vprc.vpr_correction import compute_vpr_correction

    legacy_ds, header = legacy_cor
    _, _, footer = expected_profile
    correction = compute_vpr_correction(
        processed_result.dataset,
        freezing_level_m=header.freezing_level_m,
        max_range_km=int(legacy_ds.range_km.max()),
        min_elevation_deg=header.elevation_angles[0],
        elevation_angles_deg=tuple(header.elevation_angles),
        is_precipitation=header.precip_code > 0,
        layer_top_height_m=footer.top_height,
        horizon_angle_deg=0,
        bb_result=processed_result.bright_band,
    )
    return replace(processed_result, vpr_correction=correction)


class TestVPRCorrectionComparison:
    """Compare VPR correction output against legacy .cor file.

    Legacy format outputs corrections for:
    - 2 CAPPI heights (500m, 1000m)
    - 4 elevation angles (0.7°, 1.5°, 3.0°, 3.3° for Kankaanpää)

    Uses the reference's classification and freezing level, so differences
    come from the correction calculation alone.
    """

    @pytest.mark.parametrize("cappi_height", [500, 1000])
    def test_cappi_beam_height_calculation(
        self, legacy_cor, python_profile, cappi_height
    ):
        """Compare pseudo-CAPPI heights above antenna over the full range."""
        legacy_ds, _ = legacy_cor
        assert python_profile.vpr_correction is not None
        actual = python_profile.vpr_correction.corrections["cappi_beam_height_m"].sel(
            cappi_height=cappi_height
        )
        expected = legacy_ds["beam_height_m"].sel(
            correction_type=f"cappi_{cappi_height}"
        )
        # Perl prints int() of the height.
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1)

    @pytest.mark.parametrize("cappi_height", [500, 1000])
    @pytest.mark.parametrize(
        ("python_variable", "legacy_variable"),
        [
            ("cappi_correction_db", "correction_rain_db"),
            ("cappi_clim_correction_db", "correction_klim_db"),
        ],
    )
    def test_python_cappi_corrections_match(
        self, legacy_cor, python_profile, cappi_height, python_variable, legacy_variable
    ):
        """Compare instantaneous and climatological products separately."""
        legacy_ds, _ = legacy_cor
        python_vpr = python_profile.vpr_correction
        assert python_vpr is not None and python_vpr.usable
        python_ds = python_vpr.corrections
        np.testing.assert_array_equal(python_ds.range_km, legacy_ds.range_km)
        np.testing.assert_allclose(
            python_ds[python_variable].sel(cappi_height=cappi_height),
            legacy_ds[legacy_variable].sel(correction_type=f"cappi_{cappi_height}"),
            rtol=0,
            atol=0.1,
        )

    def test_python_elevation_corrections_computed(self, processed_result):
        """Verify Python computes VPR corrections for elevation angles from VVP."""
        vpr = processed_result.vpr_correction
        assert vpr is not None, "VPR correction should be computed"

        ds = vpr.corrections

        # Should have elevation-based corrections from VVP file
        assert "elev_correction_db" in ds, (
            "Elevation-based corrections should be computed"
        )
        assert "elevation" in ds.dims, "Dataset should have elevation dimension"

        # VVP file has 3 elevations: 0.7, 1.5, 3.0
        expected_elevs = [0.7, 1.5, 3.0]
        actual_elevs = list(ds.elevation.values)
        assert actual_elevs == expected_elevs, (
            f"Elevation angles mismatch: got {actual_elevs}, expected {expected_elevs}"
        )

    def test_python_elevation_beam_heights_match_legacy(
        self, legacy_cor, python_profile
    ):
        """Fixed-elevation beam heights over the full range."""
        legacy_ds, header = legacy_cor
        python_vpr = python_profile.vpr_correction
        assert python_vpr is not None and python_vpr.usable
        python_ds = python_vpr.corrections

        for elev_idx, elev in enumerate(header.elevation_angles):
            # Perl prints int() of the height.
            np.testing.assert_allclose(
                python_ds["elev_beam_height_m"].sel(elevation=elev),
                legacy_ds["beam_height_m"].sel(correction_type=f"elev_{elev_idx + 1}"),
                rtol=0,
                atol=1,
                err_msg=f"elevation {elev}",
            )

    @pytest.mark.parametrize(
        ("python_variable", "legacy_variable"),
        [
            ("elev_correction_db", "correction_rain_db"),
            ("elev_clim_correction_db", "correction_klim_db"),
        ],
    )
    def test_python_elevation_corrections_match(
        self, legacy_cor, python_profile, python_variable, legacy_variable
    ):
        """Compare each VVP elevation against its matching reference product."""
        legacy_ds, header = legacy_cor
        vpr = python_profile.vpr_correction
        assert vpr is not None and vpr.usable
        ds = vpr.corrections
        np.testing.assert_array_equal(ds.range_km, legacy_ds.range_km)
        for elevation in ds.elevation.values:
            reference_index = header.elevation_angles.index(float(elevation)) + 1
            np.testing.assert_allclose(
                ds[python_variable].sel(elevation=elevation),
                legacy_ds[legacy_variable].sel(
                    correction_type=f"elev_{reference_index}"
                ),
                rtol=0,
                atol=0.1,
            )


def _process_like_legacy(stem: str, radar_code: str):
    """Process an input with the freezing level of its legacy reference."""
    expected, _, footer = parse_legacy_profile(DATA_DIR / f"{stem}.profile")
    radar = get_radar_metadata(radar_code)
    assert radar is not None
    result = process_vvp(
        DATA_DIR / f"{stem}.txt",
        freezing_level_m=footer.freezing_level + radar["antenna_height_m"],
    )
    return result, expected, footer


def _assert_profile_matches_legacy(result, expected, footer, stem):
    """Compare all reference levels and the lowest layer with the reference.

    Select rather than reindex the Python result so that a dropped level
    fails instead of matching MDS.
    """
    dataset = result.dataset.copy()
    # allprof_prodx2.pl prints the raw value as $mds when ZCnt <= $kynnys.
    dataset["lin_dbz"] = dataset.lin_dbz.where(dataset.zcount > MIN_SAMPLES, MDS)
    heights = expected.height.where(expected.height <= dataset.height.max(), drop=True)
    for variable in ("lin_dbz", "zcount"):
        np.testing.assert_array_equal(
            dataset[variable].sel(height=heights).values,
            expected[variable].sel(height=heights).values,
            err_msg=f"{variable} does not match legacy output for {stem}",
        )
    # The reference prints corrected dBZ with one decimal.
    np.testing.assert_allclose(
        dataset.corrected_dbz.sel(height=heights).values,
        expected.corrected_dbz.sel(height=heights).values,
        rtol=0,
        atol=0.051,
        err_msg=f"corrected_dbz does not match legacy output for {stem}",
    )
    classification = result.classification
    layer = classification.lowest_layer
    assert layer is not None
    assert (layer.bottom_height, layer.top_height, layer.has_clutter) == (
        footer.bottom_height,
        footer.top_height,
        bool(footer.ground_clutter),
    )
    assert layer.max_dbz == pytest.approx(footer.max_dbz, abs=0.051)
    assert classification.usable_for_vpr == (footer.precipitation > 0)
    assert (
        classification.precipitation_type == LEGACY_PRECIPITATION[footer.precipitation]
    )
    if footer.precipitation > 0:
        # Perl prints 0 evaporation for profiles without precipitation.
        assert layer.evaporation_amount == pytest.approx(footer.evaporation, abs=0.051)
    assert result.bright_band.detected == bool(footer.bright_band)
    assert result.vpr_correction is not None
    # The observed-profile quality, printed with two decimals. The result's
    # quality_weight falls back to the climatology weight for compositing.
    assert result.vpr_correction.corrections.attrs["quality_weight"] == pytest.approx(
        footer.quality_weight, abs=0.005
    )


# Legacy sade code: 0=none, 1=untyped, 2=snow, 3=sleet, 4=rain.
LEGACY_PRECIPITATION = {
    0: PrecipitationType.UNKNOWN,
    1: PrecipitationType.UNKNOWN,
    2: PrecipitationType.SNOW,
    3: PrecipitationType.SLEET,
    4: PrecipitationType.RAIN,
}


def test_kan_profile_matches_legacy():
    """Kankaanpää 2025-08-24 11 UTC: snow with ground clutter, no bright band."""
    stem = "202508241100_KAN.VVP_40"
    result, expected, footer = _process_like_legacy(stem, "KAN")
    _assert_profile_matches_legacy(result, expected, footer, stem)


@pytest.mark.parametrize(
    ("stem", "radar_code"),
    [
        ("202208171100_VIM.VVP_40", "VIM"),
        ("202208171245_VIM.VVP_40", "VIM"),
        # Reference generated by running allprof_prodx2.pl on the input.
        ("202611112000_KES.VVP_40", "KES"),
    ],
)
def test_levels_absent_from_input_match_legacy_profile(stem, radar_code):
    """Profiles with levels absent from the VVP file keep the legacy grid.

    Perl stores levels by height, so absent levels are output as no-echo
    levels and may be filled by smoothing.
    """
    result, expected, footer = _process_like_legacy(stem, radar_code)
    assert result.dataset.attrs["missing_levels_m"]
    _assert_profile_matches_legacy(result, expected, footer, stem)
    _assert_cappi_corrections_match_legacy(result, stem)


@pytest.mark.parametrize(
    "stem",
    [
        "202208171115_VIM.VVP_40",
        "202208171130_VIM.VVP_40",
        "202208171200_VIM.VVP_40",
        "202208171215_VIM.VVP_40",
        "202208171230_VIM.VVP_40",
    ],
)
def test_layers_reaching_ten_kilometres_match_legacy_profile(stem):
    """Perl analyses levels below 10 km above the antenna and closes layers there."""
    result, expected, footer = _process_like_legacy(stem, "VIM")
    assert int(result.dataset.height.max()) == 9902
    _assert_profile_matches_legacy(result, expected, footer, stem)
    _assert_cappi_corrections_match_legacy(result, stem)


# Kankaanpää 2025-08-24 11 UTC with one change each; references from running
# allprof_prodx2.pl and pystycappi.pl without a HIRLAM freezing level.
@pytest.mark.parametrize(
    "stem",
    [
        # ZCnt 20 at 500 m ASL: the clutter cleanup must keep the lowest level.
        "variant_KAN_level2_low_zcnt.VVP_40",
        # 300 m ASL row removed: the profile still starts at 126 m.
        "variant_KAN_no_lowest_level.VVP_40",
    ],
)
def test_near_surface_variants_match_legacy_profile_and_corrections(stem):
    """Lowest-layer classification drives the correction, as in the reference."""
    result, expected, footer = _process_like_legacy(stem, "KAN")
    _assert_profile_matches_legacy(result, expected, footer, stem)
    _assert_cappi_corrections_match_legacy(result, stem)


def _assert_cappi_corrections_match_legacy(result, stem, cappi_heights=(500, 1000)):
    """Compare CAPPI corrections with the reference .cor file.

    Unlike TestVPRCorrectionComparison, use Python's own classification
    rather than the reference's.
    """
    from vprc.vpr_correction import compute_vpr_correction

    from .legacy_parser import parse_legacy_cor

    legacy_cor, header = parse_legacy_cor(DATA_DIR / f"{stem}.cor")
    layer = result.classification.lowest_layer
    # pystycappi.pl stores each level at its 25 m bin, int(h / 25), shifting
    # the profile down by up to 24 m. Python keeps exact heights; compare on
    # the binned heights so that the shift does not mask other differences.
    dataset = result.dataset.assign_coords(height=result.dataset.height // 25 * 25)
    corrections = compute_vpr_correction(
        dataset,
        freezing_level_m=header.freezing_level_m,
        max_range_km=int(legacy_cor.range_km.max()),
        min_elevation_deg=header.elevation_angles[0],
        elevation_angles_deg=tuple(header.elevation_angles),
        is_precipitation=result.classification.usable_for_vpr,
        layer_top_height_m=layer.top_height,
        horizon_angle_deg=0,
        bb_result=result.bright_band,
    ).corrections
    products = [("cappi_clim_correction_db", "correction_klim_db")]
    if header.precip_code > 0:
        products.append(("cappi_correction_db", "correction_rain_db"))
    for python_variable, legacy_variable in products:
        for cappi_height in cappi_heights:
            # $max_dBZ_korjaus caps only positive corrections; Python caps
            # both signs (MAX_CORRECTION_DB).
            np.testing.assert_allclose(
                corrections[python_variable].sel(cappi_height=cappi_height),
                legacy_cor[legacy_variable]
                .sel(correction_type=f"cappi_{cappi_height}")
                .clip(min=-MAX_CORRECTION_DB),
                rtol=0,
                atol=0.1,
                err_msg=f"{python_variable} at {cappi_height} m for {stem}",
            )


# Operational inputs with a positive HIRLAM freezing level; references from
# running allprof_prodx2.pl and pystycappi.pl with the archived freezing level.
@pytest.mark.parametrize(
    ("stem", "radar_code", "rejection_reason"),
    [
        # Clear-air layer: no bright band search, but the no-BB clutter step.
        ("202512051200_KAN.VVP_40", "KAN", BBRejectionReason.NOT_PRECIPITATION),
        # Lowest-level peak failing the outer gradient check as well.
        ("202605052000_UTA.VVP_40", "UTA", BBRejectionReason.NEAR_SURFACE_ZCOUNT),
    ],
)
def test_positive_freezing_level_matches_legacy(stem, radar_code, rejection_reason):
    """Bright band search and the following clutter corrections match Perl."""
    result, expected, footer = _process_like_legacy(stem, radar_code)
    assert footer.freezing_level > 0
    assert result.bright_band.rejection_reason == rejection_reason
    assert result.bright_band.clutter_correction_applied
    _assert_profile_matches_legacy(result, expected, footer, stem)
    _assert_cappi_corrections_match_legacy(result, stem)


def test_bright_band_with_steep_drop_above_peak_matches_legacy():
    """Vimpeli 2026-05-05 06 UTC has a distinct band peaking at 302 m.

    The below-peak test as written in Perl also counted the 10 dB drop above
    the peak and rejected this band; Perl as run found it. Missing it changed
    the corrections near the radar by more than 20 dB.
    """
    stem = "202605050600_VIM.VVP_40"
    result, expected, footer = _process_like_legacy(stem, "VIM")
    assert footer.freezing_level > 0
    assert result.bright_band.detected
    assert result.bright_band.peak_height == 302
    assert result.bright_band.refined_peak_height == pytest.approx(
        footer.bb_height, abs=1
    )
    assert result.bright_band.bottom_height == footer.bb_bottom
    assert result.bright_band.top_height == footer.bb_top
    _assert_profile_matches_legacy(result, expected, footer, stem)
    # At 1000 m, 6 and 15 km differ by 0.06-0.12 dB (0.15 dB after Perl's
    # output rounding): Perl truncates the beam centre from 802 m to 801 m,
    # and reflectivity falls 40 dB between 400 and 700 m. With an 801 m
    # centre Python reproduces Perl's values; see compute_beam_height().
    _assert_cappi_corrections_match_legacy(result, stem, cappi_heights=(500,))


def test_filled_absent_level_does_not_bound_the_bright_band():
    """Kankaanpää 2026-04-25 03 UTC lacks 726 m above the antenna inside rain.

    With its MDS placeholder as the raw value, the band bottom snapped to the
    gap and the low-band clutter step drove the lowest levels to −42…−44 dBZ.
    Perl reads the gap as 0 dBZ; Python uses the smoothed value instead, so
    only that level differs from the reference.
    """
    stem = "202604250300_KAN.VVP_40"
    result, expected, footer = _process_like_legacy(stem, "KAN")
    absent = result.dataset.attrs["missing_levels_m"]
    assert 726 in absent
    assert result.bright_band.detected
    assert result.bright_band.refined_peak_height == pytest.approx(
        footer.bb_height, abs=1
    )
    assert result.bright_band.bottom_height == footer.bb_bottom
    assert result.bright_band.top_height == footer.bb_top
    heights = [h for h in expected.height.values if h not in absent and h <= 1926]
    np.testing.assert_allclose(
        result.dataset.corrected_dbz.sel(height=heights),
        expected.corrected_dbz.sel(height=heights),
        atol=0.1,
        rtol=0.01,
    )
    # Perl's ground reference ($korarvo in korjausprof) was -13.2 dBZ.
    assert result.vpr_correction.z_ground_dbz == pytest.approx(-13.2, abs=0.1)
