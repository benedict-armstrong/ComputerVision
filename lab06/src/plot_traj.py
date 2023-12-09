import os
from matplotlib import pyplot as plt
import numpy as np


def plot_traj(name: str, a_priori, a_posteriori, truth, out_dir, width, height):

    # Plot both
    plt.figure()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, width)
    plt.ylim(0, height)

    # invert y axis
    plt.gca().invert_yaxis()

    plt.plot(truth[:, 0], truth[:, 1], label="truth")
    plt.plot(a_priori[:, 0], a_priori[:, 1], label="a priori")
    plt.plot(a_posteriori[:, 0], a_posteriori[:, 1], label="a posteriori")
    plt.legend()

    plt.title(f"{name} trajectory")

    plt.savefig(os.path.join(
        out_dir, f"{name.replace('/', '_').replace(' ', '_')}_traj.png"))
    plt.close()
