import os
import cv2

import numpy as np
from make_gif import make_gif
from plot_dist import plot_dist
from plot_traj import plot_traj

from run_with_data import condensation_tracker_non_interactive


def run(video_path: str, out_dir, params: dict, truth_path: str = None):
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

        make_gif(video_path, out_dir, positions=pre,
                 truth=truth, bbox=params["bbox"])

        # save distance plots
        pre_dist_avg, post_dist_avg, pre_dist_length, post_dist_length = plot_dist(
            f"{video_name} pre/post", pre, post, truth, out_dir)

        vidcap = cv2.VideoCapture(video_path)
        success, frame = vidcap.read()
        frame_height, frame_width, _ = frame.shape
        vidcap.release()

        plot_traj(f"{video_name} pre/post", pre,
                  post, truth, out_dir, frame_width, frame_height)

    params["pre_dist_avg"] = pre_dist_avg
    params["post_dist_avg"] = post_dist_avg
    params["pre_dist_length"] = pre_dist_length
    params["post_dist_length"] = post_dist_length
    params["pre"] = pre
    params["post"] = post

    # save params
    with open(os.path.join(out_dir, f'{video_name}.txt'), 'w') as f:
        for key, value in params.items():
            # make a values are aligned to the right
            f.write(f"{key:<20}: {value}\n")

    return (pre_dist_avg + post_dist_avg) / 2


def run_many():
    """
    Run many experiments
    """

    distances = []
    video = "video1"
    # params = {
    #     "draw_plots": 1,
    #     "hist_bin": 16,
    #     # color histogram update parameter (0 = no update)
    #     "alpha": 0.5,
    #     "sigma_observe": 0.3,  # messurement noise
    #     "model": 1,  # system model (0 = no motion, 1 = constant velocity)
    #     "num_particles": 50,  # number of particles
    #     "sigma_position": 2,  # positional noise
    #     "sigma_velocity": 2,  # velocity noise
    #     # initial velocity  (x, y) to set particles to
    #     "initial_velocity": (0, 0),
    #     "bbox": (129, 93, 13, 13),  # (x, y, width, height) for video1
    #     "bbox": (8, 70, 13, 13),  # (x, y, width, height) for video2
    #     "bbox": (22, 84, 13, 13),  # (x, y, width, height) for video3
    # }

    params = {
        "draw_plots": 1,
        "hist_bin": 16,
        "alpha": 0.2,  # color histogram update parameter (0 = no update)
        "sigma_observe": 0.2,  # messurement noise
        "model": 0,  # system model (0 = no motion, 1 = constant velocity)
        "num_particles": 50,
        "sigma_position": 10,  # positional noise
        "sigma_velocity": 5,  # velocity noise
        # initial velocity  (x, y) to set particles to
        "initial_velocity": (10, 0),
        "bbox": (129, 93, 13, 13),  # (x, y, width, height) for video1
    }
    variying_param = "hist_bin"
    values = [6, 8, 16, 32, 64]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, .9, 1]

    for v in values:
        params[variying_param] = v

        print(f"Running {video} with {variying_param} = {v}")

        d = run(f'./data/{video}.avi', f'out/{video}/{variying_param}/{v}/',
                params, f'data/{video}_truth.npy')

        print(f"Distance: {d}")

        distances.append(d)

    if len(values) > 1:
        # plot distances
        import matplotlib.pyplot as plt
        plt.plot(values, distances)
        plt.xlabel(variying_param)
        plt.ylabel("Distance")

        # y axis log scale
        # plt.yscale("log")

        plt.title(f"Distance vs {variying_param} for {video}")

        plt.savefig(
            f"out/{video}/{variying_param}/distances_{variying_param}.png")
        plt.show()
        plt.close()


if __name__ == "__main__":
    run_many()
