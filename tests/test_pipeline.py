"""Regression tests for the public VVP processing pipeline."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from vprc import process_vvp
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
    assert np.isfinite(result.vpr_correction.corrections.cappi_blended_correction_db).all()
