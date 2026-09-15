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
import pytest
import xarray as xr

from vprc import process_vvp, average_corrections
from vprc.temporal import MIN_AGE_WEIGHT
from vprc.vpr_correction import VPRCorrectionResult

from .legacy_parser import parse_legacy_cor, parse_legacy_corave


# Paths to test data
DATA_DIR = Path(__file__).parent / 'data'

# VIM radar VVP files (7 timesteps)
VIM_FILES = sorted(DATA_DIR.glob('202208171*_VIM.VVP_40.txt'))

# Expected legacy outputs
VIM_COR_FILES = sorted(DATA_DIR.glob('202208171*_VIM.VVP_40.cor'))
VIM_CORAVE = DATA_DIR / '202208171245_VIM.VVP_40.corave'


class TestTemporalAveragingE2E:
    """End-to-end tests comparing Python temporal averaging with legacy output."""

    def test_data_files_exist(self):
        """Verify all required test data files exist."""
        assert len(VIM_FILES) == 7, f"Expected 7 VIM VVP files, found {len(VIM_FILES)}"
        assert len(VIM_COR_FILES) == 7, f"Expected 7 VIM .cor files, found {len(VIM_COR_FILES)}"
        assert VIM_CORAVE.exists(), f"Expected .corave file not found: {VIM_CORAVE}"

    def test_process_all_vim_profiles(self):
        """Verify all VIM profiles can be processed."""
        for vvp_file in VIM_FILES:
            result = process_vvp(vvp_file)
            assert result.vpr_correction is not None, f"VPR correction failed for {vvp_file}"
            assert result.vpr_correction.usable, f"VPR not usable for {vvp_file}"

    def test_average_corrections_produces_result(self):
        """Verify temporal averaging produces a valid result."""
        results = []
        for vvp_file in VIM_FILES:
            result = process_vvp(vvp_file)
            if result.vpr_correction and result.vpr_correction.usable:
                results.append(result.vpr_correction)

        assert len(results) == 7, "All 7 profiles should produce usable VPR corrections"

        averaged = average_corrections(results)
        assert averaged.usable
        assert 'cappi_correction_db' in averaged.corrections

    def test_averaged_corrections_reasonable_values(self):
        """Verify averaged corrections are in reasonable range."""
        results = []
        for vvp_file in VIM_FILES:
            result = process_vvp(vvp_file)
            if result.vpr_correction:
                results.append(result.vpr_correction)

        averaged = average_corrections(results)

        # Corrections should be in reasonable dB range
        cappi_corr = averaged.corrections['cappi_correction_db']
        assert np.all(cappi_corr >= -30), "Corrections should not be below -30 dB"
        assert np.all(cappi_corr <= 30), "Corrections should not exceed 30 dB"

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

    def test_sample_count_matches(self):
        """Verify sample counts from processed profiles."""
        results = []
        for vvp_file in VIM_FILES:
            result = process_vvp(vvp_file)
            if result.vpr_correction:
                results.append(result.vpr_correction)

        # Should have 7 profiles
        assert len(results) == 7

        # Legacy shows count=7 for all ranges
        legacy_ds, _ = parse_legacy_corave(VIM_CORAVE)
        assert np.all(legacy_ds['count_klim'] == 7)
        assert np.all(legacy_ds['count_rain'] == 7)


class TestLegacyWeightingComparison:
    """Tests specifically comparing the weighting schemes."""

    def test_legacy_index_weighting(self):
        """Verify we understand legacy index-based weighting.

        Legacy Perl uses: weight[j] = (j+1) / n
        For 7 profiles: weights = [1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 7/7]
        """
        n = 7
        legacy_weights = np.array([(j + 1) / n for j in range(n)])

        # Sum of weights
        weight_sum = legacy_weights.sum()
        expected_sum = sum(range(1, n + 1)) / n  # = 28/7 = 4.0
        assert weight_sum == pytest.approx(expected_sum)

        # Normalized weights
        normalized = legacy_weights / weight_sum
        assert normalized.sum() == pytest.approx(1.0)

    def test_python_timedelta_weighting_equally_spaced(self):
        """Verify Python time-delta weighting for equally-spaced data.

        Python uses: weight = 1 - age_fraction * (1 - MIN_AGE_WEIGHT)
        For equally-spaced data, this creates different weights than Perl.
        """
        from vprc.temporal import _compute_age_weights

        # Create 7 equally-spaced timestamps (15 min apart)
        base = datetime(2022, 8, 17, 11, 0)
        from datetime import timedelta
        timestamps = [base + timedelta(minutes=15 * i) for i in range(7)]

        py_weights = _compute_age_weights(timestamps)

        # Oldest: MIN_AGE_WEIGHT, newest: 1.0
        assert py_weights[0] == pytest.approx(MIN_AGE_WEIGHT)
        assert py_weights[-1] == pytest.approx(1.0)

        # Linear progression for equally spaced
        expected = np.linspace(MIN_AGE_WEIGHT, 1.0, 7)
        np.testing.assert_allclose(py_weights, expected)

    def test_weighting_difference_quantified(self):
        """Quantify the difference between legacy and Python weighting.

        For n=7 equally-spaced profiles, compare the two schemes.
        """
        n = 7

        # Legacy: index-based
        legacy_weights = np.array([(j + 1) / n for j in range(n)])
        legacy_normalized = legacy_weights / legacy_weights.sum()

        # Python: time-delta based (equally spaced = linear)
        py_weights = np.linspace(MIN_AGE_WEIGHT, 1.0, n)
        py_normalized = py_weights / py_weights.sum()

        # Compare normalized weights
        weight_diff = np.abs(legacy_normalized - py_normalized)

        # Maximum difference in normalized weights
        max_diff = weight_diff.max()

        # The differences should be modest (< 10%)
        assert max_diff < 0.1, f"Weight difference too large: {max_diff:.3f}"

        # Python weights give more emphasis to newest
        # Legacy: newest has weight 7/28 = 0.25 of total
        # Python: newest has weight 1.0/(0.2+0.33+0.47+0.60+0.73+0.87+1.0) ≈ 0.24
        print(f"Legacy normalized weights: {legacy_normalized}")
        print(f"Python normalized weights: {py_normalized}")
        print(f"Max absolute difference: {max_diff:.4f}")


class TestCorParserValidation:
    """Validate that we can parse legacy .cor files correctly."""

    def test_parse_vim_cor_file(self):
        """Verify .cor file parsing."""
        cor_file = VIM_COR_FILES[0]
        ds, header = parse_legacy_cor(cor_file)

        assert header.radar_name == 'VIMPELI'
        assert header.product == 'VVP_40'
        assert len(header.elevation_angles) == 4
        assert header.elevation_angles == [0.3, 0.7, 1.5, 3.0]

        # Check data structure
        assert 'correction_klim_db' in ds
        assert 'correction_rain_db' in ds
        assert 'beam_height_m' in ds

        # Range should go 1-251 km (legacy includes 251)
        assert ds.range_km.min() == 1
        assert ds.range_km.max() == 251

    def test_parse_vim_corave_file(self):
        """Verify .corave file parsing."""
        ds, header = parse_legacy_corave(VIM_CORAVE)

        assert header.radar_name == 'VIMPELI'
        assert header.product == 'VVP_40'
        assert len(header.elevation_angles) == 4

        # Check data structure
        assert 'correction_avg_db' in ds
        assert 'correction_rain_db' in ds
        assert 'count_klim' in ds
        assert 'count_rain' in ds

        # Sample counts should all be 7 (7 input files)
        assert np.all(ds['count_klim'] == 7)
        assert np.all(ds['count_rain'] == 7)
