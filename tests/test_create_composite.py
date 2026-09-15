"""Failure handling in the composite creation command."""

import runpy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner


SCRIPT = Path(__file__).parents[1] / "src" / "scripts" / "create_composite.py"
KAN_FILE = Path(__file__).parent / "data" / "202308281000_KAN.VVP_40.txt"


@pytest.fixture
def command():
    return runpy.run_path(str(SCRIPT))["main"]


@pytest.mark.parametrize("error", [ValueError("invalid profile"), OSError("unreadable")])
def test_input_error_does_not_create_success_product(command, tmp_path, error):
    output = tmp_path / "output"
    with patch.dict(
        command.callback.__globals__,
        process_vvp=Mock(side_effect=error),
    ):
        result = CliRunner().invoke(
            command, [str(KAN_FILE), "--output-dir", str(output)]
        )
    assert result.exit_code != 0
    assert str(error) in result.output
    assert "no precipitation" not in result.output
    assert not output.exists()


def test_unexpected_error_propagates(command, tmp_path):
    error = RuntimeError("programming error")
    output = tmp_path / "output"
    with patch.dict(
        command.callback.__globals__,
        process_vvp=Mock(side_effect=error),
    ):
        result = CliRunner().invoke(
            command, [str(KAN_FILE), "--output-dir", str(output)]
        )
    assert result.exception is error
    assert not output.exists()


def test_unknown_radar_is_an_error(command, tmp_path):
    path = tmp_path / "202508241100_XXX.VVP_40.txt"
    path.write_text(KAN_FILE.read_text())
    output = tmp_path / "output"

    result = CliRunner().invoke(command, [str(path), "--output-dir", str(output)])

    assert result.exit_code != 0
    assert "No coordinates configured" in result.output
    assert not output.exists()
