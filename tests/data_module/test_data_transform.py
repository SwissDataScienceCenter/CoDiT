import jax.numpy as jnp
import pytest

from codit.data_module.data_transform import (
    haversine_distance,
    inverse_haversine,
    velocity_vector_from_pos_time,
)


class TestHaversineDistance:
    def test_same_point_gives_zero(self):
        pos = jnp.array([48.8566, 2.3522])
        dist = haversine_distance(pos, pos)
        assert float(dist) == pytest.approx(0.0, abs=1e-3)

    def test_one_degree_latitude_at_equator(self):
        """1 degree of latitude ≈ 111,195 m anywhere on Earth."""
        pos1 = jnp.array([0.0, 0.0])
        pos2 = jnp.array([1.0, 0.0])
        dist = haversine_distance(pos1, pos2)
        assert float(dist) == pytest.approx(111_195.0, rel=0.01)

    def test_one_degree_longitude_at_equator(self):
        """1 degree of longitude at equator ≈ 111,195 m."""
        pos1 = jnp.array([0.0, 0.0])
        pos2 = jnp.array([0.0, 1.0])
        dist = haversine_distance(pos1, pos2)
        assert float(dist) == pytest.approx(111_195.0, rel=0.01)

    def test_symmetry(self):
        pos1 = jnp.array([48.8566, 2.3522])
        pos2 = jnp.array([51.5074, -0.1278])
        assert float(haversine_distance(pos1, pos2)) == pytest.approx(
            float(haversine_distance(pos2, pos1)), rel=1e-5
        )

    def test_returns_scalar(self):
        pos = jnp.array([0.0, 0.0])
        dist = haversine_distance(pos, pos)
        assert dist.shape == ()


class TestInverseHaversine:
    def test_zero_distance_returns_same_position(self):
        pos = jnp.array([48.8566, 2.3522])
        new_pos = inverse_haversine(pos, 0.0, 0.0)
        assert jnp.allclose(new_pos, pos, atol=1e-6)

    def test_round_trip_100km_north(self):
        """Move 100 km north, measure haversine from origin to new position ≈ 100 km."""
        pos = jnp.array([0.0, 0.0])
        bearing = 0.0  # north
        distance_m = 100_000.0
        delta = inverse_haversine(pos, distance_m, bearing)
        new_pos = pos + delta
        recovered_dist = haversine_distance(pos, new_pos)
        assert float(recovered_dist) == pytest.approx(distance_m, rel=0.01)

    def test_returns_2d_vector(self):
        pos = jnp.array([0.0, 0.0])
        delta = inverse_haversine(pos, 1000.0, 90.0)
        assert delta.shape == (2,)


class TestVelocityVectorFromPosTime:
    def test_stationary_gives_zero_velocity(self):
        pos = jnp.array([0.0, 0.0])
        vel = velocity_vector_from_pos_time(pos, pos, time_in_seconds=3600.0)
        assert jnp.allclose(vel, jnp.zeros(2), atol=1e-6)

    def test_returns_2d_vector(self):
        pos1 = jnp.array([0.0, 0.0])
        pos2 = jnp.array([1.0, 0.0])
        vel = velocity_vector_from_pos_time(pos1, pos2, time_in_seconds=3600.0)
        assert vel.shape == (2,)

    def test_speed_is_positive_for_moving_points(self):
        pos1 = jnp.array([0.0, 0.0])
        pos2 = jnp.array([1.0, 0.0])
        vel = velocity_vector_from_pos_time(pos1, pos2, time_in_seconds=3600.0)
        speed = jnp.linalg.norm(vel)
        assert float(speed) > 0.0
