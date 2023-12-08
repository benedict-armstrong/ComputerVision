import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib import patches


def draw(video_name, positions: np.array, truth: np.array, bbox, name=None):
    '''
    video_name - video name

    '''
    # Choose video
    if video_name == "video1.avi":
        first_frame = 10
        last_frame = 42
    elif video_name == "video2.avi":
        first_frame = 3
        last_frame = 40
    elif video_name == "video3.avi":
        first_frame = 1
        last_frame = 60

    # Change this to where your data is
    data_dir = './data/'
    video_path = os.path.join(data_dir, video_name)

    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(1, first_frame)

    bbox_width = bbox[2]
    bbox_height = bbox[3]
    top_left = (bbox[0], bbox[1])
    bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])

    fig, ax = plt.subplots(1)
    ret, frame = vidcap.read()
    im = ax.imshow(frame)

    distances = []

    plt.ion()

    for i in range(last_frame - first_frame):
        t = i + first_frame

        # Get frame
        ret, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        im.set_data(frame)
        to_remove = []

        pos_rect = ax.add_patch(patches.Rectangle((positions[i, 0] - 0.5 * bbox_width, positions[i, 1] - 0.5 * bbox_height),
                                                  bbox_width, bbox_height, fill=False, edgecolor='blue', lw=2))

        truth_rect = ax.add_patch(patches.Rectangle((truth[i, 0] - 0.5 * bbox_width, truth[i, 1] - 0.5 * bbox_height),
                                                    bbox_width, bbox_height, fill=False, edgecolor='green', lw=1))

        distance = np.linalg.norm(positions[i, :] - truth[i, :])
        distances.append(distance)

        to_remove.append(pos_rect)
        to_remove.append(truth_rect)
        ax.set_title(f"Frame: {t}, Distance: {distance:.2f}")

        # =====================================

        if t != last_frame:

            plt.pause(0.1)
            # Remove previous element from plot
            for e in to_remove:
                e.remove()

    plt.ioff()

    # Plot distance
    plt.figure()
    plt.plot(distances)
    average_distance = np.mean(distances)
    plt.title(
        f"Distance (avg: {average_distance:.2f}, {video_name}, {bbox_width}x{bbox_height} bbox)")
    plt.xlabel("Frame")
    plt.ylabel("Distance")
    plt.savefig(
        f"out/{video_name.split('.')[0]}{f'_{name}' if name else ''}_distance.png")


if __name__ == "__main__":
    video_name = 'video3.avi'
    data = np.load(f'out/{video_name.split(".")[0]}.npz')

    pre = data["pre"]
    post = data["post"]
    bbox = data["bbox"]

    truth = np.load(f'out/{video_name.split(".")[0]}_truth.npy')

    draw(video_name, positions=pre, truth=truth, bbox=bbox, name="pre")
