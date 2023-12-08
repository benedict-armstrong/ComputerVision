import numpy as np


def resample(
    particles: np.array,
    particles_w: np.array
):
    """
    This function should resample the particles based on their weights
    """
    n = particles.shape[0]
    new_particles = particles[
        np.random.choice(n, n, p=particles_w.flatten())]
    new_particles_w = np.ones(particles_w.shape) / particles_w.shape[0]

    return new_particles, new_particles_w
