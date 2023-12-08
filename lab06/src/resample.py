import numpy as np


def resample(
    particles: np.array,
    particles_w: np.array
):
    """
    This function should resample the particles based on their weights
    """
    n = particles.shape[0]
    idx = np.random.choice(n, n, p=particles_w.flatten())

    new_particles = particles[idx, :]

    new_particles_w = particles_w[idx, :]
    new_particles_w = new_particles_w / np.sum(new_particles_w)

    return new_particles, new_particles_w
