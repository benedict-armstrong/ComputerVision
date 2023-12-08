import numpy as np
from numpy.testing import assert_almost_equal


def estimate(
    particles: np.array,
    particles_w: np.array
):
    """
    this function should estimate the mean state given the particles and their weights
    """
    mean_state = np.zeros(particles.shape[-1])
    # compute the weighted mean of all particles should have shape [2,]

    temp = particles * particles_w

    assert np.sum(particles_w) != 0
    assert_almost_equal(np.sum(particles_w), 1)

    mean_state = np.sum(temp, axis=0)

    return mean_state
