"""Tests for the temporal averaging module."""

from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from vprc.temporal import average_corrections, _extract_timestamps, _compute_age_weights, MIN_AGE_WEIGHT
from vprc.vpr_correction import VPRCorrectionResult


def make_test_correction(
    correction_value: float = 1.0,
    z_ground_dbz: float = 30.0,
    timestamp: datetime | None = None,
    range_km: list[int] | None = None,
    cappi_heights: list[int] | None = None,
    elevation_angles: list[float] | None = None,
    usable: bool = True,
) -> VPRCorrectionResult:
    """Create a VPRCorrectionResult for testing.

    All corrections are set to `correction_value` for easy verification.
    """
    if timestamp is None:
        timestamp = datetime(2025, 8, 24, 11, 0)
    if range_km is None:
        range_km = [1, 2, 3, 4, 5]
    if cappi_heights is None:
        cappi_heights = [500, 1000]

    n_range = len(range_km)
    n_heights = len(cappi_heights)

    data_vars = {
        "cappi_correction_db": (
            ["range_km", "cappi_height"],
            np.full((n_range, n_heights), correction_value),
        ),
        "cappi_beam_height_m": (
            ["range_km", "cappi_height"],
            np.full((n_range, n_heights), 500.0),
        ),
    }
    coords = {
        "range_km": range_km,
        "cappi_height": cappi_heights,
    }

    if elevation_angles is not None:
        n_elevs = len(elevation_angles)
        data_vars["elev_correction_db"] = (
            ["range_km", "elevation"],
            np.full((n_range, n_elevs), correction_value),
        )
        data_vars["elev_beam_height_m"] = (
            ["range_km", "elevation"],
            np.full((n_range, n_elevs), 500.0),
        )
        coords["elevation"] = elevation_angles

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "radar": "TEST",
            "timestamp": timestamp.isoformat(),
            "z_ground_dbz": z_ground_dbz,
        },
    )

    return VPRCorrectionResult(
        corrections=ds,
        z_ground_dbz=z_ground_dbz,
        usable=usable,
    )


class TestAverageCorrections:
    """Tests for average_corrections function."""

    def test_single_profile_unchanged(self):
        """Single profile averaging returns identical result."""
        result = make_test_correction(correction_value=5.0, z_ground_dbz=25.0)

        averaged = average_corrections([result])

        # Should return the same result
        assert averaged.z_ground_dbz == 25.0
        assert averaged.usable is True
        np.testing.assert_array_equal(
            averaged.corrections["cappi_correction_db"].values,
            result.corrections["cappi_correction_db"].values,
        )

    def test_two_profiles_equal_values(self):
        """Two profiles with same values average to that value."""
        t1 = datetime(2025, 8, 24, 10, 0)
        t2 = datetime(2025, 8, 24, 11, 0)

        r1 = make_test_correction(correction_value=3.0, timestamp=t1)
        r2 = make_test_correction(correction_value=3.0, timestamp=t2)

        averaged = average_corrections([r1, r2])

        np.testing.assert_allclose(
            averaged.corrections["cappi_correction_db"].values,
            3.0,
        )

    def test_two_profiles_weighted_average(self):
        """Two profiles are averaged with time-delta weights."""
        t1 = datetime(2025, 8, 24, 10, 0)  # older
        t2 = datetime(2025, 8, 24, 11, 0)  # newer

        # Older profile: 0 dB, newer: 3 dB
        r1 = make_test_correction(correction_value=0.0, timestamp=t1)
        r2 = make_test_correction(correction_value=3.0, timestamp=t2)

        averaged = average_corrections([r1, r2])

        # Weights: oldest=MIN_AGE_WEIGHT=0.2, newest=1.0, sum=1.2
        # Average = (0*0.2 + 3*1.0) / 1.2 = 2.5
        expected = 2.5
        np.testing.assert_allclose(
            averaged.corrections["cappi_correction_db"].values,
            expected,
        )

    def test_three_profiles_weighted_average(self):
        """Three equally-spaced profiles get linear time-delta weights."""
        base = datetime(2025, 8, 24, 10, 0)

        r1 = make_test_correction(correction_value=0.0, timestamp=base)
        r2 = make_test_correction(
            correction_value=3.0, timestamp=base + timedelta(hours=1)
        )
        r3 = make_test_correction(
            correction_value=6.0, timestamp=base + timedelta(hours=2)
        )

        averaged = average_corrections([r1, r2, r3])

        # Weights: oldest=0.2, middle=0.6, newest=1.0 (linear interpolation)
        # Sum = 0.2 + 0.6 + 1.0 = 1.8
        # Average = (0*0.2 + 3*0.6 + 6*1.0) / 1.8 = (0 + 1.8 + 6) / 1.8 ≈ 4.33
        expected = (0 * 0.2 + 3 * 0.6 + 6 * 1.0) / 1.8
        np.testing.assert_allclose(
            averaged.corrections["cappi_correction_db"].values,
            expected,
        )

    def test_unsorted_timestamps_sorted(self):
        """Profiles are sorted by timestamp before averaging."""
        base = datetime(2025, 8, 24, 10, 0)

        # Pass in wrong order
        r2 = make_test_correction(
            correction_value=3.0, timestamp=base + timedelta(hours=1)
        )
        r1 = make_test_correction(correction_value=0.0, timestamp=base)
        r3 = make_test_correction(
            correction_value=6.0, timestamp=base + timedelta(hours=2)
        )

        averaged = average_corrections([r2, r1, r3])  # wrong order

        # Should produce same result as sorted
        expected = (0 * 0.2 + 3 * 0.6 + 6 * 1.0) / 1.8
        np.testing.assert_allclose(
            averaged.corrections["cappi_correction_db"].values,
            expected,
        )

    def test_irregular_spacing_weights_by_time(self):
        """Irregular spacing is handled correctly."""
        base = datetime(2025, 8, 24, 10, 0)

        # Big gap then small gap: 0h, 9h, 10h
        r1 = make_test_correction(correction_value=0.0, timestamp=base)
        r2 = make_test_correction(
            correction_value=10.0, timestamp=base + timedelta(hours=9)
        )
        r3 = make_test_correction(
            correction_value=20.0, timestamp=base + timedelta(hours=10)
        )

        averaged = average_corrections([r1, r2, r3])

        # Weights: oldest=0.2, middle=0.92 (9/10 of way), newest=1.0
        w = [0.2, 0.92, 1.0]
        expected = (0 * w[0] + 10 * w[1] + 20 * w[2]) / sum(w)
        np.testing.assert_allclose(
            averaged.corrections["cappi_correction_db"].values,
            expected,
        )

    def test_z_ground_dbz_averaged(self):
        """z_ground_dbz is also time-weighted averaged."""
        t1 = datetime(2025, 8, 24, 10, 0)
        t2 = datetime(2025, 8, 24, 11, 0)

        r1 = make_test_correction(z_ground_dbz=20.0, timestamp=t1)
        r2 = make_test_correction(z_ground_dbz=30.0, timestamp=t2)

        averaged = average_corrections([r1, r2])

        # Weights: 0.2, 1.0, sum=1.2
        # Average = (20*0.2 + 30*1.0) / 1.2 = 34/1.2 ≈ 28.33
        expected = (20 * 0.2 + 30 * 1.0) / 1.2
        assert averaged.z_ground_dbz == pytest.approx(expected)

    def test_elevation_corrections_included(self):
        """Elevation-based corrections are also averaged."""
        t1 = datetime(2025, 8, 24, 10, 0)
        t2 = datetime(2025, 8, 24, 11, 0)

        r1 = make_test_correction(
            correction_value=0.0, timestamp=t1, elevation_angles=[0.5, 1.5]
        )
        r2 = make_test_correction(
            correction_value=3.0, timestamp=t2, elevation_angles=[0.5, 1.5]
        )

        averaged = average_corrections([r1, r2])

        assert "elev_correction_db" in averaged.corrections
        # Weights: 0.2, 1.0 -> (0*0.2 + 3*1.0) / 1.2 = 2.5
        np.testing.assert_allclose(
            averaged.corrections["elev_correction_db"].values,
            2.5,
        )

    def test_usable_if_any_usable(self):
        """Result is usable if at least one input is usable."""
        t1 = datetime(2025, 8, 24, 10, 0)
        t2 = datetime(2025, 8, 24, 11, 0)

        r1 = make_test_correction(timestamp=t1, usable=False)
        r2 = make_test_correction(timestamp=t2, usable=True)

        averaged = average_corrections([r1, r2])
        assert averaged.usable is True

    def test_not_usable_if_none_usable(self):
        """Result is not usable if no inputs are usable."""
        t1 = datetime(2025, 8, 24, 10, 0)
        t2 = datetime(2025, 8, 24, 11, 0)

        r1 = make_test_correction(timestamp=t1, usable=False)
        r2 = make_test_correction(timestamp=t2, usable=False)

        averaged = average_corrections([r1, r2])
        assert averaged.usable is False

    def test_attrs_from_newest(self):
        """Output attrs come from newest profile."""
        t1 = datetime(2025, 8, 24, 10, 0)
        t2 = datetime(2025, 8, 24, 11, 0)

        r1 = make_test_correction(timestamp=t1)
        r1.corrections.attrs["radar"] = "OLD_RADAR"

        r2 = make_test_correction(timestamp=t2)
        r2.corrections.attrs["radar"] = "NEW_RADAR"

        averaged = average_corrections([r1, r2])

        assert averaged.corrections.attrs["radar"] == "NEW_RADAR"
        assert averaged.corrections.attrs["timestamp"] == t2.isoformat()
        assert averaged.corrections.attrs["averaged_from_n"] == 2

    def test_empty_list_raises(self):
        """Empty results list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            average_corrections([])

    def test_explicit_timestamps(self):
        """Explicit timestamps parameter overrides attrs."""
        t1 = datetime(2025, 8, 24, 10, 0)
        t2 = datetime(2025, 8, 24, 11, 0)

        # Create with wrong timestamps in attrs
        r1 = make_test_correction(
            correction_value=0.0, timestamp=datetime(2000, 1, 1)
        )
        r2 = make_test_correction(
            correction_value=3.0, timestamp=datetime(2000, 1, 2)
        )

        # Override with explicit timestamps
        averaged = average_corrections([r1, r2], timestamps=[t1, t2])

        # Should compute weighted average with new weights: 0.2, 1.0
        # (0*0.2 + 3*1.0) / 1.2 = 2.5
        np.testing.assert_allclose(
            averaged.corrections["cappi_correction_db"].values,
            2.5,
        )
        # Output timestamp should be latest from explicit list
        assert averaged.corrections.attrs["timestamp"] == t2.isoformat()

    def test_mismatched_timestamps_length_raises(self):
        """Mismatched timestamps length raises ValueError."""
        r1 = make_test_correction()
        r2 = make_test_correction()

        with pytest.raises(ValueError, match="must match"):
            average_corrections([r1, r2], timestamps=[datetime.now()])


class TestExtractTimestamps:
    """Tests for _extract_timestamps helper."""

    def test_extracts_valid_timestamps(self):
        """Valid ISO timestamps are extracted."""
        t1 = datetime(2025, 8, 24, 10, 0)
        t2 = datetime(2025, 8, 24, 11, 0)

        r1 = make_test_correction(timestamp=t1)
        r2 = make_test_correction(timestamp=t2)

        timestamps = _extract_timestamps([r1, r2])

        assert timestamps == [t1, t2]

    def test_missing_timestamp_raises(self):
        """Missing timestamp attribute raises ValueError."""
        result = make_test_correction()
        del result.corrections.attrs["timestamp"]

        with pytest.raises(ValueError, match="missing 'timestamp'"):
            _extract_timestamps([result])

    def test_invalid_timestamp_raises(self):
        """Invalid timestamp string raises ValueError."""
        result = make_test_correction()
        result.corrections.attrs["timestamp"] = "not-a-date"

        with pytest.raises(ValueError, match="invalid timestamp"):
            _extract_timestamps([result])


class TestComputeAgeWeights:
    """Tests for _compute_age_weights helper."""

    def test_single_timestamp(self):
        """Single timestamp returns weight 1.0."""
        timestamps = [datetime(2025, 8, 24, 10, 0)]
        weights = _compute_age_weights(timestamps)

        np.testing.assert_array_equal(weights, [1.0])

    def test_two_timestamps_equal_spacing(self):
        """Two timestamps give oldest=MIN_AGE_WEIGHT, newest=1.0."""
        t1 = datetime(2025, 8, 24, 10, 0)
        t2 = datetime(2025, 8, 24, 11, 0)
        weights = _compute_age_weights([t1, t2])

        np.testing.assert_allclose(weights, [MIN_AGE_WEIGHT, 1.0])

    def test_three_timestamps_equal_spacing(self):
        """Three equally-spaced timestamps give linear weights."""
        base = datetime(2025, 8, 24, 10, 0)
        timestamps = [
            base,
            base + timedelta(hours=1),
            base + timedelta(hours=2),
        ]
        weights = _compute_age_weights(timestamps)

        # oldest=0.2, middle=0.6, newest=1.0
        np.testing.assert_allclose(weights, [0.2, 0.6, 1.0])

    def test_irregular_spacing_respects_time_gaps(self):
        """Irregular spacing weights by actual time position."""
        base = datetime(2025, 8, 24, 10, 0)
        # Large gap at start, small gap at end
        timestamps = [
            base,                          # oldest
            base + timedelta(hours=9),     # close to newest
            base + timedelta(hours=10),    # newest
        ]
        weights = _compute_age_weights(timestamps)

        # oldest: age_fraction=1.0 → weight=0.2
        # middle: age_fraction=0.1 → weight=1.0 - 0.1*0.8 = 0.92
        # newest: age_fraction=0.0 → weight=1.0
        np.testing.assert_allclose(weights, [0.2, 0.92, 1.0])

    def test_identical_timestamps_equal_weights(self):
        """Identical timestamps get equal weight 1.0."""
        t = datetime(2025, 8, 24, 10, 0)
        timestamps = [t, t, t]
        weights = _compute_age_weights(timestamps)

        np.testing.assert_array_equal(weights, [1.0, 1.0, 1.0])
