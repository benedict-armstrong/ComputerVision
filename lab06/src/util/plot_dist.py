import os
from matplotlib import pyplot as plt
import numpy as np


def plot_dist(name: str, a_priori, a_posteriori, truth, out_dir):
    distances1 = a_priori - truth
    distances1 = np.linalg.norm(distances1, axis=1)

    distances2 = a_posteriori - truth
    distances2 = np.linalg.norm(distances2, axis=1)

    # Plot both distances on the same plot with a legend
    plt.figure()

    plt.plot(distances1, label="a priori")
    plt.plot(distances2, label="a posteriori")
    plt.legend()

    plt.xlabel("Frame")
    plt.ylabel("Distance (log)")
    plt.yscale("log")

    m1, m2, d1, d2 = np.mean(distances1), np.mean(
        distances2), np.linalg.norm(distances1), np.linalg.norm(distances2)

    plt.title(
        f"{name} dist. avg: {m1:.2f}/{m2:.2f}, norm: {d1:.2f}/{d2:.2f}")

    plt.savefig(os.path.join(
        out_dir, f"{name.replace('/', '_').replace(' ', '_')}_distance.png"))
    plt.close()

    return m1, m2, d1, d2
