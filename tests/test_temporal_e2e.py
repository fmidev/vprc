"""End-to-end tests for temporal averaging against legacy Perl output.

These tests validate that the Python temporal averaging implementation
produces results compatible with pystycappi_ka.pl.

Test data: tests/data/202208171*_VIM.VVP_40.* files
- .txt: Input VVP profiles (7 timesteps: 11:00-12:45, missing 11:45)
- .cor: Individual VPR corrections from pystycappi.pl
- .corave: Time-averaged corrections from pystycappi_ka.pl

Important notes:
- Default Python weighting uses actual elapsed time.
- Opt-in legacy weighting uses rank: weight = (j+1)/n.
- Tests isolate temporal arithmetic using the checked-in instantaneous
  corrections, comparing all products/ranges against the averaged reference.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from vprc import average_corrections, process_vvp
from vprc.temporal import MIN_AGE_WEIGHT
from vprc.vpr_correction import VPRCorrectionResult

from .legacy_parser import parse_legacy_cor, parse_legacy_corave

# Paths to test data
DATA_DIR = Path(__file__).parent / "data"

# VIM radar VVP files (7 timesteps)
VIM_FILES = sorted(DATA_DIR.glob("202208171*_VIM.VVP_40.txt"))

# Expected legacy outputs
VIM_COR_FILES = sorted(DATA_DIR.glob("202208171*_VIM.VVP_40.cor"))
VIM_CORAVE = DATA_DIR / "202208171245_VIM.VVP_40.corave"


class TestTemporalAveragingE2E:
    """End-to-end tests comparing Python temporal averaging with legacy output."""

    def test_data_files_exist(self):
        """Verify all required test data files exist."""
        assert len(VIM_FILES) == 7, f"Expected 7 VIM VVP files, found {len(VIM_FILES)}"
        assert len(VIM_COR_FILES) == 7, (
            f"Expected 7 VIM .cor files, found {len(VIM_COR_FILES)}"
        )
        assert VIM_CORAVE.exists(), f"Expected .corave file not found: {VIM_CORAVE}"

    def test_numerical_temporal_average_against_legacy(self):
        """Match every blended and rain-only value within output rounding."""
        results = []
        for path in VIM_COR_FILES:
            ds, header = parse_legacy_cor(path)
            blended = (
                ds.correction_klim_db * 0.2
                + ds.correction_rain_db * header.quality_weight
            ) / (0.2 + header.quality_weight)
            corrections = xr.Dataset(
                {
                    "correction_avg_db": blended,
                    "correction_rain_db": ds.correction_rain_db,
                },
                attrs={"timestamp": header.timestamp.isoformat()},
            )
            results.append(VPRCorrectionResult(corrections, z_ground_dbz=0.0))

        averaged = average_corrections(results, weighting="legacy").corrections
        expected, _ = parse_legacy_corave(VIM_CORAVE)
        for name in ("correction_avg_db", "correction_rain_db"):
            np.testing.assert_allclose(
                averaged[name].sel(range_km=expected.range_km),
                expected[name],
                atol=0.051,
                rtol=0,
            )

    def test_pipeline_results_are_weighted_by_their_own_timestamps(self):
        """Weights follow each profile's timestamp, including the 11:45 gap."""
        results = []
        for path in VIM_FILES:
            correction = process_vvp(path).vpr_correction
            assert correction is not None and correction.usable
            results.append(correction)
        times = np.array(
            [datetime.strptime(path.name[:12], "%Y%m%d%H%M") for path in VIM_FILES]
        )
        age_fraction = (times[-1] - times) / (times[-1] - times[0])
        weights = 1 - age_fraction * (1 - MIN_AGE_WEIGHT)

        # Input order must not matter.
        averaged = average_corrections(results[::-1])

        for name in ("cappi_blended_correction_db", "elev_blended_correction_db"):
            expected = (
                sum(
                    w * r.corrections[name]
                    for w, r in zip(weights, results, strict=True)
                )
                / weights.sum()
            )
            xr.testing.assert_allclose(averaged.corrections[name], expected)
        assert averaged.corrections.attrs["timestamp"] == times[-1].isoformat()
