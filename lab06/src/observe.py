import numpy as np
from chi2_cost import chi2_cost
from numpy.testing import assert_almost_equal
from color_histogram import color_histogram


def observe(
    particles: np.array,
    frame: np.array,
    bbox_height: int,
    bbox_width: int,
    hist_bins: int,
    hist_target: np.array,
    sigma_observe: float
):
    """

    """

    particles_w = np.zeros(particles.shape[0])
    for i in range(len(particles)):

        x, y, *_ = particles[i]

        hist = color_histogram(
            top_x=int(max(0, round(x - 0.5 * bbox_width))),
            top_y=int(max(0, round(y - 0.5 * bbox_height))),
            bottom_x=int(min(frame.shape[1], round(x + 0.5 * bbox_width))),
            bottom_y=int(min(frame.shape[0], round(y + 0.5 * bbox_height))),
            image=frame,
            hist_bin=hist_bins
        )

        distance = chi2_cost(hist, hist_target)
        particles_w[i] = np.exp(-distance / (2 * sigma_observe ** 2))

    if np.sum(particles_w) == 0:
        # print("Warning: all weights are zero")

        # return as shape (N, 1) uniform weights
        # this means we have probably lost the target
        particles_w = np.ones(particles_w.shape[0]) / particles_w.shape[0]

        # one thing we could do is to reinitialize the particles with the prevoius weights
        # particles_w = particles_w_prev
    else:
        # return as shape (N, 1) normalized weights
        particles_w = particles_w / np.sum(particles_w)

    assert_almost_equal(np.sum(particles_w), 1)

    return particles_w[:, np.newaxis]
