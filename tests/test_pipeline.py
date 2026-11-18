"""Regression tests for the public VVP processing pipeline."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from vprc import BBRejectionReason, LayerType, PrecipitationType, process_vvp
from vprc.config import _load_radar_defaults
from vprc.vpr_correction import compute_vpr_correction

KAN_FILE = Path(__file__).parent / "data" / "202508241100_KAN.VVP_40.txt"


@pytest.mark.parametrize("freezing_asl", [100.0, 2000.0])
def test_correction_uses_above_antenna_freezing_level(freezing_asl):
    result = process_vvp(KAN_FILE, freezing_level_m=freezing_asl)
    freezing_above_antenna = max(0, freezing_asl - 174)

    assert result.dataset.attrs["freezing_level_m"] == freezing_above_antenna
    assert result.classification.freezing_level_m == freezing_above_antenna
    correction = result.vpr_correction
    assert correction is not None
    assert correction.corrections.attrs["freezing_level_m"] == freezing_above_antenna
    expected = compute_vpr_correction(
        result.dataset,
        freezing_level_m=freezing_above_antenna,
        bb_result=result.bright_band,
        is_precipitation=result.usable_for_vpr,
        layer_top_height_m=result.classification.lowest_layer.top_height,
        min_elevation_deg=result.dataset.attrs["elevation_angles"][0],
    )
    xr.testing.assert_allclose(correction.corrections, expected.corrections)


def test_configured_freezing_level_is_used_without_detected_fallback(monkeypatch):
    config = _load_radar_defaults()
    configured = {**config, "KAN": {**config["KAN"], "freezing_level_m": 2000}}
    monkeypatch.setattr("vprc.config._load_radar_defaults", lambda: configured)

    result = process_vvp(KAN_FILE, fallback_detected_freezing_level=False)

    assert result.vpr_correction is not None
    assert result.dataset.attrs["freezing_level_m"] == 1826
    assert result.vpr_correction.corrections.attrs["freezing_level_m"] == 1826


def test_explicit_freezing_level_overrides_configuration(monkeypatch):
    config = _load_radar_defaults()
    configured = {**config, "KAN": {**config["KAN"], "freezing_level_m": 2000}}
    monkeypatch.setattr("vprc.config._load_radar_defaults", lambda: configured)

    result = process_vvp(KAN_FILE, freezing_level_m=0)

    assert result.vpr_correction is not None
    assert result.vpr_correction.corrections.attrs["freezing_level_m"] == 0
    assert np.isfinite(
        result.vpr_correction.corrections.cappi_blended_correction_db
    ).all()


def _write_profile(
    tmp_path, values, *, wind_count=1000, zcount=1000, elevations="0.7 1.5 3.0"
):
    path = tmp_path / "202508241100_KAN.VVP_40.txt"
    lines = [
        f"KANKAANPAA VVP_40 PPI1_A 2025 08 24 11 00 ELEVS: {elevations}",
        "",
        "Height Count ZCnt ...",
    ]
    for i, value in enumerate(values):
        lines.append(
            f"{100 + 200 * i} {wind_count} {zcount} 0 0 0 0 0 0 {value} "
            "0 0 0 0 0 0 0 0 0"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def test_correction_uses_lowest_available_elevation(tmp_path):
    path = _write_profile(
        tmp_path,
        [35, 30, 25, 20, 18, 16, 14, 12, 10, 8, 6, 4],
        elevations="1.5 0.3 0.7",
    )

    result = process_vvp(
        path, antenna_height_m=0, lowest_level_offset_m=100, freezing_level_m=1000
    )

    assert result.vpr_correction is not None
    assert result.vpr_correction.corrections.attrs["min_elevation_deg"] == 0.3


def test_surface_bright_band_reaches_correction_pipeline(tmp_path):
    values = [35, 30, 25, 20, 18, 16, 14, 12, 10, 8, 6, 4]
    path = _write_profile(tmp_path, values)

    result = process_vvp(
        path, antenna_height_m=0, lowest_level_offset_m=100, freezing_level_m=300
    )

    assert result.bright_band.detected
    assert result.bright_band.peak_height == 100
    assert result.bright_band.bottom_height == -1
    assert result.bright_band.top_height == 700
    assert result.classification.precipitation_type == PrecipitationType.SLEET
    assert result.dataset.corrected_dbz.sel(height=100).item() == 30
    assert result.vpr_correction is not None
    assert result.vpr_correction.z_ground_dbz == 20


def test_bb_enhancement_is_not_evaporation(tmp_path):
    values = [10, 10, 10, 12, 18, 30, 35, 35, 30, 18, 10, 8, 6, 4, 2, 0]
    path = _write_profile(tmp_path, values)

    result = process_vvp(
        path, antenna_height_m=0, lowest_level_offset_m=100, freezing_level_m=1500
    )

    assert result.bright_band.detected
    assert result.classification.profile_type == LayerType.PRECIPITATION
    assert result.classification.lowest_layer.evaporation_amount == 8
    assert not result.classification.lowest_layer.evaporation_detected
    assert result.vpr_correction.quality_weight > 0


def test_wind_count_does_not_mask_reflectivity(tmp_path):
    values = [35, 30, 25, 20, 18, 16, 14, 12, 10, 8, 6, 4]
    path = _write_profile(tmp_path, values, wind_count=0)

    result = process_vvp(
        path, antenna_height_m=0, lowest_level_offset_m=100, freezing_level_m=300
    )

    assert result.bright_band.detected
    assert result.usable_for_vpr
    assert result.vpr_correction.quality_weight > 0


def test_evaporation_rejects_observed_correction_after_bb_processing(tmp_path):
    values = [-20, -20, -20, -10, 0, 10, 18, 30, 35, 35, 30, 18, 10, 8, 6, 4, 2, 0]
    path = _write_profile(tmp_path, values)

    result = process_vvp(
        path, antenna_height_m=0, lowest_level_offset_m=100, freezing_level_m=1900
    )

    assert result.classification.lowest_layer.evaporation_detected
    assert result.classification.lowest_layer.evaporation_amount == 38
    assert result.classification.profile_type == LayerType.ALTOSTRATUS
    assert not result.bright_band.detected
    assert result.bright_band.rejection_reason == BBRejectionReason.EVAPORATION
    assert result.classification.precipitation_type == PrecipitationType.UNKNOWN
    assert result.vpr_correction.corrections.attrs["climatology_only"]


def test_nonprecipitation_uses_only_climatology(tmp_path):
    path = _write_profile(tmp_path, [20] * 8)

    result = process_vvp(
        path, antenna_height_m=0, lowest_level_offset_m=100, freezing_level_m=1000
    )

    assert not result.usable_for_vpr
    assert result.classification.profile_type == LayerType.CLEAR_AIR_ECHO
    assert result.classification.precipitation_type == PrecipitationType.UNKNOWN
    correction = result.vpr_correction
    assert correction is not None
    assert correction.corrections.attrs["quality_weight"] == 0
    assert correction.corrections.attrs["climatology_only"]
    xr.testing.assert_allclose(
        correction.corrections.cappi_blended_correction_db,
        correction.corrections.cappi_clim_correction_db,
    )
