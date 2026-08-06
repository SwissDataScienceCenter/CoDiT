import jax.numpy as jnp
import pytest

from codit.data_module.data_transform import (
    haversine_distance,
    inverse_haversine,
    inverse_haversine_grad,
    transform_traj_to_metre,
    velocity_to_distance_bearing,
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
        """Move 100 km north from Paris, verify haversine distance ≈ 100 km."""
        pos = jnp.array([48.8566, 2.3522])  # Paris
        bearing = 0.0  # north
        distance_m = 100_000.0
        new_pos = inverse_haversine(pos, distance_m, bearing)
        recovered_dist = haversine_distance(pos, new_pos)
        assert float(recovered_dist) == pytest.approx(distance_m, rel=0.01)

    def test_returns_absolute_position_not_delta(self):
        """Verify that inverse_haversine returns absolute position, not delta."""
        pos = jnp.array([48.8566, 2.3522])  # Paris
        result = inverse_haversine(pos, 100_000.0, 0.0)
        # Absolute position must be far from origin (not near-zero like a delta)
        assert float(jnp.linalg.norm(result)) > 1.0

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


class TestTransformTrajToMetre:
    def test_returns_correct_shape(self):
        T = 11
        traj = jnp.zeros((T, 2))
        result = transform_traj_to_metre(traj)
        assert result.shape == (T, 2)

    def test_stationary_gives_zero(self):
        T = 5
        traj = jnp.tile(jnp.array([48.8566, 2.3522]), (T, 1))
        result = transform_traj_to_metre(traj)
        assert jnp.allclose(result, jnp.zeros((T, 2)), atol=1e-3)

    def test_moving_trajectory_nonzero(self):
        """A trajectory moving 1 degree north per step should give nonzero displacements."""
        T = 4
        lats = jnp.arange(T, dtype=jnp.float32)
        traj = jnp.stack([lats, jnp.zeros(T)], axis=-1)
        result = transform_traj_to_metre(traj)
        # All rows after the first should have nonzero lat-component
        assert jnp.all(jnp.abs(result[1:, 0]) > 0.0)

    def test_first_row_is_zero(self):
        """Displacement from traj[0] to traj[0] must be zero for any trajectory."""
        traj = jnp.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]])
        result = transform_traj_to_metre(traj)
        assert jnp.allclose(result[0], jnp.zeros(2), atol=1e-3)


class TestVelocityToDistanceBearing:
    def test_zero_velocity_gives_zero_distance(self):
        speed, _ = velocity_to_distance_bearing(jnp.array([0.0, 0.0]))
        assert float(speed) == pytest.approx(0.0, abs=1e-6)

    def test_speed_matches_norm(self):
        v = jnp.array([3.0, 4.0])
        speed, _ = velocity_to_distance_bearing(v)
        assert float(speed) == pytest.approx(5.0, rel=1e-6)

    def test_east_velocity_gives_zero_bearing(self):
        """velocity_vector = (vx, vy). arctan2(vy=0, vx=1) → 0°."""
        _, bearing = velocity_to_distance_bearing(jnp.array([1.0, 0.0]))
        assert float(bearing) == pytest.approx(0.0, abs=1e-6)

    def test_north_velocity_gives_90_bearing(self):
        """arctan2(vy=1, vx=0) → 90°."""
        _, bearing = velocity_to_distance_bearing(jnp.array([0.0, 1.0]))
        assert float(bearing) == pytest.approx(90.0, abs=1e-6)


class TestHaversineReferenceDistances:
    def test_paris_london_distance(self):
        """Great-circle distance Paris ↔ London ≈ 344 km (public reference)."""
        paris = jnp.array([48.8566, 2.3522])
        london = jnp.array([51.5074, -0.1278])
        dist = haversine_distance(paris, london)
        assert float(dist) == pytest.approx(344_000.0, rel=0.02)

    def test_antipodes_distance_half_circumference(self):
        """Antipodal points are ≈ π·R apart on a sphere."""
        pos1 = jnp.array([0.0, 0.0])
        pos2 = jnp.array([0.0, 180.0])
        expected = jnp.pi * 6_371_000.0
        assert float(haversine_distance(pos1, pos2)) == pytest.approx(
            float(expected), rel=1e-4
        )


class TestInverseHaversineBearings:
    def test_east_bearing_increases_longitude(self):
        """Bearing 90° (east) → the new longitude should exceed the starting one."""
        pos = jnp.array([0.0, 0.0])
        new_pos = inverse_haversine(pos, 100_000.0, 90.0)
        assert float(new_pos[1]) > float(pos[1])

    def test_south_bearing_decreases_latitude(self):
        pos = jnp.array([10.0, 0.0])
        new_pos = inverse_haversine(pos, 100_000.0, 180.0)
        assert float(new_pos[0]) < float(pos[0])

    def test_north_100km_reaches_expected_latitude(self):
        """100 km north ≈ 0.899° latitude increase (100 km / 111.195 km-per-deg)."""
        pos = jnp.array([0.0, 0.0])
        new_pos = inverse_haversine(pos, 100_000.0, 0.0)
        assert float(new_pos[0]) == pytest.approx(0.8993, rel=0.01)

    def test_grad_returns_delta_not_absolute(self):
        """inverse_haversine_grad returns (dlat, dlon), not absolute coords."""
        pos = jnp.array([48.8566, 2.3522])
        delta = inverse_haversine_grad(pos, 100_000.0, 0.0)
        # A ~100km north delta is < 1 degree; absolute lat would be > 40 degrees
        assert float(jnp.abs(delta[0])) < 5.0

    def test_grad_and_absolute_sum_to_new_position(self):
        """inverse_haversine == pos + inverse_haversine_grad (identity by construction)."""
        pos = jnp.array([48.8566, 2.3522])
        d, bearing = 50_000.0, 45.0
        absolute = inverse_haversine(pos, d, bearing)
        delta = inverse_haversine_grad(pos, d, bearing)
        assert jnp.allclose(absolute, pos + delta, atol=1e-4)


class TestVelocityVectorScaling:
    def test_doubling_time_halves_speed(self):
        """Fixed endpoints, 2× time → ½ speed."""
        pos1 = jnp.array([0.0, 0.0])
        pos2 = jnp.array([0.5, 0.0])
        v_fast = velocity_vector_from_pos_time(pos1, pos2, 3600.0)
        v_slow = velocity_vector_from_pos_time(pos1, pos2, 7200.0)
        assert float(jnp.linalg.norm(v_slow)) == pytest.approx(
            0.5 * float(jnp.linalg.norm(v_fast)), rel=1e-5
        )
