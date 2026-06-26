"""Tests for data transformations."""

import numpy as np
import pytest
from marine_local_mtgnn.data.transforms import (
    degrees_to_components,
    components_to_degrees,
    apply_log_transform,
    reverse_log_transform,
)


class TestDirectionConversion:
    """Test direction/speed ↔ u/v component conversions."""

    def test_north_direction_from(self):
        """Test north direction (0°) with FROM convention."""
        direction = np.array([0.0])
        speed = np.array([10.0])

        u, v = degrees_to_components(direction, speed, convention="from")

        # From north (0°) means wind comes from north, points south
        assert np.isclose(u[0], 0, atol=1e-10)
        assert np.isclose(v[0], -10, atol=1e-10)

    def test_east_direction_from(self):
        """Test east direction (90°) with FROM convention."""
        direction = np.array([90.0])
        speed = np.array([10.0])

        u, v = degrees_to_components(direction, speed, convention="from")

        # From east (90°) means wind comes from east, points west
        assert np.isclose(u[0], -10, atol=1e-10)
        assert np.isclose(v[0], 0, atol=1e-10)

    def test_north_direction_to(self):
        """Test north direction (0°) with TO convention."""
        direction = np.array([0.0])
        speed = np.array([10.0])

        u, v = degrees_to_components(direction, speed, convention="to")

        # To north (0°) means vector points north
        assert np.isclose(u[0], 0, atol=1e-10)
        assert np.isclose(v[0], 10, atol=1e-10)

    def test_round_trip_from(self):
        """Test round-trip conversion for FROM convention."""
        directions_in = np.array([0, 45, 90, 180, 270, 360])
        speeds_in = np.array([5, 10, 15, 8, 12, 20])

        u, v = degrees_to_components(directions_in, speeds_in, convention="from")
        directions_out, speeds_out = components_to_degrees(u, v, convention="from")

        np.testing.assert_allclose(directions_out % 360, directions_in % 360, atol=1e-10)
        np.testing.assert_allclose(speeds_out, speeds_in, atol=1e-10)

    def test_round_trip_to(self):
        """Test round-trip conversion for TO convention."""
        directions_in = np.array([0, 45, 90, 180, 270])
        speeds_in = np.array([5, 10, 15, 8, 12])

        u, v = degrees_to_components(directions_in, speeds_in, convention="to")
        directions_out, speeds_out = components_to_degrees(u, v, convention="to")

        np.testing.assert_allclose(directions_out % 360, directions_in % 360, atol=1e-10)
        np.testing.assert_allclose(speeds_out, speeds_in, atol=1e-10)

    def test_zero_speed(self):
        """Test with zero speed (edge case)."""
        direction = np.array([0, 90, 180])
        speed = np.array([0, 0, 0])

        u, v = degrees_to_components(direction, speed, convention="from")

        np.testing.assert_allclose(u, 0, atol=1e-10)
        np.testing.assert_allclose(v, 0, atol=1e-10)


class TestLogTransforms:
    """Test log and log1p transforms."""

    def test_log_transform(self):
        """Test log transform for wave parameters."""
        data = np.array([1, 10, 100])
        result = apply_log_transform(data, "significant_wave_height_m")

        expected = np.array([0, np.log(10), np.log(100)])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_log_transform_very_small(self):
        """Test log transform handles very small values."""
        data = np.array([0, 1e-10, 1e-5])
        result = apply_log_transform(data, "significant_wave_height_m")

        # Should not raise, should be finite
        assert np.all(np.isfinite(result))
        # Smallest value should be around log(1e-6)
        assert result[0] < -10

    def test_log1p_transform(self):
        """Test log1p transform for radiation parameters."""
        data = np.array([0, 100, 1000])
        result = apply_log_transform(data, "global_radiation_wm2")

        expected = np.log1p(data)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_log_reverse_transform(self):
        """Test reversing log transform."""
        data = np.array([1, 10, 100])
        transformed = apply_log_transform(data, "significant_wave_height_m")
        recovered = reverse_log_transform(transformed, "significant_wave_height_m")

        np.testing.assert_allclose(recovered, data, rtol=1e-10)

    def test_log1p_reverse_transform(self):
        """Test reversing log1p transform."""
        data = np.array([0, 100, 1000])
        transformed = apply_log_transform(data, "global_radiation_wm2")
        recovered = reverse_log_transform(transformed, "global_radiation_wm2")

        np.testing.assert_allclose(recovered, data, rtol=1e-10)

    def test_identity_for_non_log_param(self):
        """Test that non-log parameters are returned unchanged."""
        data = np.array([1.5, 2.5, 3.5])
        result = apply_log_transform(data, "air_temp_c")

        np.testing.assert_allclose(result, data, atol=1e-10)
