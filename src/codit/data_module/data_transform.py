import jax
import jax.numpy as jnp


def velocity_to_distance_bearing(
    velocity_vector: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Calculate the distance and bearing from a velocity vector using JAX.

    Parameters:
        velocity_vector: Tuple or list containing the velocity components (vx, vy) in m/s.

    Returns:
        Tuple containing the distance traveled and bearing (in degrees).
    """
    # Convert velocity vector components to JAX array
    velocity_array = jnp.array(velocity_vector)

    # Calculate distance traveled using the magnitude of the velocity vector
    speed = jnp.linalg.norm(velocity_array)

    # Calculate bearing (direction of motion) in radians
    bearing_rad = jnp.arctan2(velocity_array[1], velocity_array[0])

    # Convert bearing from radians to degrees
    bearing_deg = jnp.degrees(bearing_rad)

    return speed, bearing_deg


def inverse_haversine(pos: tuple | jax.Array, distance: float | jax.Array, bearing: float | jax.Array) -> jax.Array:
    """
    Calculate the new latitude and longitude given a starting point,
    distance (in meters), and bearing (in degrees) using JAX.

    Parameters:
        lat, lon: Latitude and longitude of the starting point (in degrees)
        distance: Distance in meters
        bearing: Bearing in degrees (0 degrees is north, 90 degrees is east, etc.)

    Returns:
        Tuple containing the new latitude and longitude.
    """
    # Radius of the Earth in meters
    R = 6371000.0
    lat, lon = pos[0], pos[1]
    # Convert latitude and longitude from degrees to radians
    lat, lon, bearing = jnp.radians(jnp.array([lat, lon, bearing]))

    # Calculate new latitude and longitude
    new_lat = jnp.arcsin(
        jnp.sin(lat) * jnp.cos(distance / R)
        + jnp.cos(lat) * jnp.sin(distance / R) * jnp.cos(bearing)
    )
    new_lon = lon + jnp.arctan2(
        jnp.sin(bearing) * jnp.sin(distance / R) * jnp.cos(lat),
        jnp.cos(distance / R) - jnp.sin(lat) * jnp.sin(new_lat),
    )

    # Convert new latitude and longitude from radians to degrees
    new_lat, new_lon = jnp.degrees(new_lat), jnp.degrees(new_lon)

    return jnp.array([new_lat, new_lon])


def velocity_vector_from_pos_time(
    pos_start: jax.Array, pos_end: jax.Array, time_in_seconds: float
) -> jax.Array:
    """
    Calculate the velocity vector required to reach the end position from the start position with constant velocity.

    Parameters:
        pos_start: Latitude and longitude of the start position (in degrees).
        pos_end: Latitude and longitude of the end position (in degrees).
        time_seconds: Time in seconds.

    Returns:
        Velocity vector (vx, vy) in meters per second.
    """
    pos_start_rad = jnp.radians(pos_start)
    pos_end_rad = jnp.radians(pos_end)
    dpos = pos_end_rad - pos_start_rad
    distance_in_metres = haversine_distance(pos_start, pos_end)
    velocity_magnitude = distance_in_metres / time_in_seconds
    bearing_rad = jnp.arctan2(
        jnp.sin(dpos[1]) * jnp.cos(pos_end_rad[0]),
        jnp.cos(pos_start_rad[0]) * jnp.sin(pos_end_rad[0])
        - jnp.sin(pos_start_rad[0]) * jnp.cos(pos_end_rad[0]) * jnp.cos(dpos[1]),
    )
    velocity_vector = velocity_magnitude * jnp.array(
        [jnp.cos(bearing_rad), jnp.sin(bearing_rad)]
    )
    return velocity_vector


def inverse_haversine_grad(
    pos: jax.Array, distance: float | jax.Array, bearing: float | jax.Array
) -> jax.Array:
    """
    Calculate the new latitude and longitude given a starting point,
    distance (in meters), and bearing (in degrees) using JAX.

    Parameters:
        lat, lon: Latitude and longitude of the starting point (in degrees)
        distance: Distance in meters
        bearing: Bearing in degrees (0 degrees is north, 90 degrees is east, etc.)

    Returns:
        Tuple containing the new latitude and longitude.
    """
    # Radius of the Earth in meters
    R = 6371000.0
    lat_deg, lon_deg = pos[0], pos[1]
    # Convert latitude and longitude from degrees to radians
    lat_rad, lon_rad, bearing = jnp.radians(jnp.array([lat_deg, lon_deg, bearing]))

    # Calculate new latitude and longitude
    new_lat = jnp.arcsin(
        jnp.sin(lat_rad) * jnp.cos(distance / R)
        + jnp.cos(lat_rad) * jnp.sin(distance / R) * jnp.cos(bearing)
    )
    new_lon = lon_rad + jnp.arctan2(
        jnp.sin(bearing) * jnp.sin(distance / R) * jnp.cos(lat_rad),
        jnp.cos(distance / R) - jnp.sin(lat_rad) * jnp.sin(new_lat),
    )

    # Convert new latitude and longitude from radians to degrees
    new_lat, new_lon = jnp.degrees(new_lat), jnp.degrees(new_lon)
    dlat = new_lat - lat_deg
    dlon = new_lon - lon_deg
    return jnp.array([dlat, dlon])


def haversine_distance(pos1: jax.Array, pos2: jax.Array) -> jax.Array:
    """
    Calculate the great-circle distance (in meters) between two points specified by latitude and longitude.

    Parameters:
        lat1, lon1: Latitude and longitude of the first point (in degrees).
        lat2, lon2: Latitude and longitude of the second point (in degrees).

    Returns:
        Distance in meters.
    """
    # Radius of the Earth in meters
    R = 6371000.0

    # Convert latitude and longitude from degrees to radians
    pos1, pos2 = jnp.radians(pos1), jnp.radians(pos2)

    # Haversine formula
    dpos = pos2 - pos1
    a = (
        jnp.sin(dpos[0] / 2) ** 2
        + jnp.cos(pos1[0]) * jnp.cos(pos2[0]) * jnp.sin(dpos[1] / 2) ** 2
    )
    c = 2 * jnp.arctan2(jnp.sqrt(a), jnp.sqrt(1 - a))

    # Distance in meters
    distance = R * c
    return distance
