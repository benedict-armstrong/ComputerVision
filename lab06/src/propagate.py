import numpy as np


def propagate(
    particles: np.array,
    frame_height: int,
    frame_width: int,
    params: dict
):
    """
    Propagate the particles (0 = no motion, 1 = constant velocity)
    """
    model = params["model"]
    s_pos = params["sigma_position"]
    n = particles.shape[0]
    w = np.random.normal(0, s_pos, (n, 2))

    if model == 0:
        # use original position with some noise
        particles = particles + w
    elif model == 1:
        s_vel = params["sigma_velocity"]
        # we use a constant velocity model A
        velocities = particles[:, 2:]
        positions = particles[:, :2]

        # add some noise to the velocities
        w_v = np.random.normal(0, s_vel, (n, 2))
        velocities += w_v

        # add the velocities and some noise to positions
        positions += velocities + w

        # update the particle position and velocity
        particles[:, :2] = positions
        particles[:, 2:] = velocities
    else:
        raise ValueError("Unknown model")

    # we make sure that the particles are within the frame
    particles[:, 0] = np.clip(particles[:, 0], 0, frame_width)
    particles[:, 1] = np.clip(particles[:, 1], 0, frame_height)

    return particles
