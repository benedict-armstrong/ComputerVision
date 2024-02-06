import cv2
import numpy as np

from color_histogram import color_histogram
from estimate import estimate
from propagate import propagate
from observe import observe
from resample import resample


def condensation_tracker_non_interactive(video_path: str, params: dict):
    '''
    video_path - video path

    params - parameters
        - draw_plats {0, 1} draw output plots throughout
        - hist_bin   1-255 number of histogram bins for each color: proper values 4,8,16
        - alpha      number in [0,1]; color histogram update parameter (0 = no update)
        - sigma_position   std. dev. of system model position noise
        - sigma_observe    std. dev. of observation model noise
        - num_particles    number of particles
        - model            {0,1} system model (0 = no motion, 1 = constant velocity)
    if using model = 1 then the following parameters are used:
        - sigma_velocity   std. dev. of system model velocity noise
        - initial_velocity initial velocity to set particles to
        - bbox          (x_center, y_center, width, height) for initial bounding box
    '''

    pre_positions = []
    post_positions = []

    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(1, 0)
    ret, first_image = vidcap.read()

    number_of_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("number_of_frames", number_of_frames)

    frame_height = first_image.shape[0]
    frame_width = first_image.shape[1]

    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    bbox = params["bbox"]
    bbox_width = bbox[2]
    bbox_height = bbox[3]
    top_left = (round(bbox[0] - bbox_width/2), round(bbox[1] - bbox_height/2))
    bottom_right = (round(bbox[0] + bbox_width/2),
                    round(bbox[1] + bbox_height/2))

    # Get initial color histogram
    # === implement fuction color_histogram() ===
    hist = color_histogram(top_left[0], top_left[1], bottom_right[0], bottom_right[1],
                           first_image, params["hist_bin"])
    # ===========================================

    state_length = 2
    if (params["model"] == 1):
        state_length = 4

    # a priori mean state
    mean_state_a_priori = np.zeros(
        [number_of_frames, state_length])
    mean_state_a_posteriori = np.zeros(
        [number_of_frames, state_length])
    # bounding box centre
    mean_state_a_priori[0, 0:2] = [
        (top_left[0] + bottom_right[0])/2., (top_left[1] + bottom_right[1])/2.]

    if params["model"] == 1:
        # use initial velocity
        mean_state_a_priori[0, 2:4] = params["initial_velocity"]

    # Initialize Particles
    particles = np.tile(mean_state_a_priori[0], (params["num_particles"], 1))
    particles_w = np.ones([params["num_particles"], 1]) * \
        1./params["num_particles"]

    for i in range(number_of_frames + 1):

        # Propagate particles
        # === Implement function propagate() ===
        particles = propagate(particles, frame_height, frame_width, params)
        # ======================================

        # Estimate
        # === Implement function estimate() ===
        mean_state_a_priori[i, :] = estimate(particles, particles_w)
        # =====================================

        # Get frame
        ret, frame = vidcap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Observe
        # === Implement function observe() ===
        particles_w = observe(particles, frame, bbox_height, bbox_width,
                              params["hist_bin"], hist, params["sigma_observe"])
        # ====================================

        # Update estimation
        mean_state_a_posteriori[i, :] = estimate(particles, particles_w)

        # Update histogram color model
        hist_crrent = color_histogram(min(max(0, round(mean_state_a_posteriori[i, 0]-0.5*bbox_width)), frame_width-1),
                                      min(max(
                                          0, round(mean_state_a_posteriori[i, 1]-0.5*bbox_height)), frame_height-1),
                                      min(max(
                                          0, round(mean_state_a_posteriori[i, 0]+0.5*bbox_width)), frame_width-1),
                                      min(max(
                                          0, round(mean_state_a_posteriori[i, 1]+0.5*bbox_height)), frame_height-1),
                                      frame, params["hist_bin"])

        hist = (1 - params["alpha"]) * hist + params["alpha"] * hist_crrent

        # RESAMPLE PARTICLES
        # === Implement function resample() ===
        particles, particles_w = resample(particles, particles_w)
        # =====================================

        post_positions.append(
            (mean_state_a_posteriori[i, 0], mean_state_a_posteriori[i, 1]))
        pre_positions.append(
            (mean_state_a_priori[i, 0], mean_state_a_priori[i, 1]))

    return np.array(pre_positions), np.array(post_positions)


if __name__ == "__main__":
    video_path = './data/video3.avi'
    params = {
        "draw_plots": 1,
        "hist_bin": 16,
        "alpha": 0.3,  # color histogram update parameter (0 = no update)
        "sigma_observe": 0.4,  # messurement noise
        "model": 1,  # system model (0 = no motion, 1 = constant velocity)
        "num_particles": 40,
        "sigma_position": 1,  # positional noise
        "sigma_velocity": 4,  # velocity noise
        # initial velocity  (x, y) to set particles to
        "initial_velocity": (5, 0),
        "bbox": (22, 84, 13, 13),  # (x, y, width, height)
    }
    pre, post = condensation_tracker_non_interactive(
        video_path, params)

    # print(pre.shape, post.shape)
    # print(pre, post)

    # save as numpy array
    v, extension = video_path.split('.')

    np.savez(
        f'out/{v}.npz',
        pre=pre,
        post=post,
        bbox=np.array(params["bbox"]),
    )
