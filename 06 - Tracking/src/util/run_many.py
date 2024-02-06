import os
import cv2
import itertools

import numpy as np
from .make_gif import make_gif
from .plot_dist import plot_dist
from .plot_traj import plot_traj

from .run_with_data import condensation_tracker_non_interactive


def run(video_path: str, out_dir, params: dict, truth_path: str = None, save_gif: bool = True, save_dist: bool = True):
    # create output directory if not exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pre, post = condensation_tracker_non_interactive(video_path, params)

    # save data
    video_name = os.path.basename(video_path).split('.')[0]
    np.savez(
        os.path.join(out_dir, f'{video_name}.npz'),
        pre=pre,
        post=post,
        bbox=np.array(params["bbox"]),
    )

    # save gif
    if truth_path is not None:
        truth = np.load(truth_path)
        np.savez(
            os.path.join(out_dir, f'{video_name}.npz'),
            pre=pre,
            post=post,
            truth=truth,
            bbox=np.array(params["bbox"]),
        )

        if save_gif:
            make_gif(video_path, out_dir, pre=pre, post=post,
                     truth=truth, bbox=params["bbox"])

        if save_dist:
            # save distance plots
            plot_dist(f"{video_name} pre/post", pre, post, truth, out_dir)

        vidcap = cv2.VideoCapture(video_path)
        success, frame = vidcap.read()
        frame_height, frame_width, _ = frame.shape
        vidcap.release()

        plot_traj(f"{video_name} pre/post", pre,
                  post, truth, out_dir, frame_width, frame_height)

    distances1 = pre - truth
    distances1 = np.linalg.norm(distances1, axis=1)

    distances2 = post - truth
    distances2 = np.linalg.norm(distances2, axis=1)

    pre_dist_avg, post_dist_avg, pre_dist_length, post_dist_length = np.mean(distances1), np.mean(
        distances2), np.linalg.norm(distances1), np.linalg.norm(distances2)

    data = {} | params
    data["pre_dist_avg"] = pre_dist_avg
    data["post_dist_avg"] = post_dist_avg
    data["pre_dist_length"] = pre_dist_length
    data["post_dist_length"] = post_dist_length
    data["pre"] = pre
    data["post"] = post

    # save params
    with open(os.path.join(out_dir, f'{video_name}.txt'), 'w') as f:
        for key, value in data.items():
            # make a values are aligned to the right
            f.write(f"{key:<20}: {value}\n")

    return (pre_dist_avg + post_dist_avg) / 2


def run_many():
    """
    Run many experiments
    """

    video = "video2"
    params = {
        "draw_plots": 1,
        "hist_bin": 32,
        "alpha": 0.1,  # color histogram update parameter (1 = no update)
        "sigma_observe": 1.5,  # messurement noise
        "model": 1,  # system model (0 = no motion, 1 = constant velocity)
        "num_particles": 100,
        "sigma_position": 20,  # positional noise
        "sigma_velocity": 0.5,  # velocity noise
        # initial velocity  (x, y) to set particles to
        "initial_velocity": (5, -10),
        # "bbox": (136, 104, 20, 16)  # video1
        "bbox": (15, 80, 20, 20)  # video2
    }
    variying_param = {
        "model": [0, 1],
        "sigma_observe": [0.3, 0.5, 1, 1.5, 2, 4, 8],
        "repeats": [1, 1, 1],
    }
    # values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, .9, 1]

    # save distances for 3d plot
    distances = []
    vp = list(variying_param.keys())

    # test with all combinations of variying params
    for idx, v in enumerate(itertools.product(*variying_param.values())):
        for i, key in enumerate(variying_param.keys()):
            params[key] = v[i]

        print(f"Running {video} with {vp} = {v}")

        d = run(
            f'./data/cut_{video}.avi',
            f'out/{video}/{vp}/{v}/',
            params,
            f'data/{video}_truth.npy',
            save_gif=False,
            save_dist=False
        )

        print(f"Distance: {d}")

        distances.append(d)

    # plot distances
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot third dimension as color

    repeats = True if "repeats" in variying_param else False

    ax.scatter(
        [v[0] for v in itertools.product(*variying_param.values())],
        [v[1] for v in itertools.product(*variying_param.values())],
        distances,
        c=[v[2] for v in itertools.product(
            *variying_param.values())] if not repeats else 'blue',
    )

    ax.set_xlabel(vp[0])
    ax.set_ylabel(vp[1])
    ax.set_zlabel("Distance")

    if not repeats:
        # add color legend
        # https://stackoverflow.com/a/49703292/1323144
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.get_cmap('viridis'), norm=plt.Normalize(vmin=min([v[2] for v in itertools.product(*variying_param.values())]), vmax=max([v[2] for v in itertools.product(*variying_param.values())])))
        sm._A = []
        plt.colorbar(sm, ax=ax)

    plt.title(
        f"Distance vs {vp[0]}, {vp[1]}, {len(variying_param['repeats'] if repeats else vp[2])} for {video}")

    plt.savefig(
        f"out/{video}/{vp}/distances_{vp[0]}_{vp[1]}_{vp[2]}.png")
    plt.show()
    # plt.close()


if __name__ == "__main__":
    run_many()
