import cv2
import os
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Rectangle


def make_gif(video_path: str, out_dir, positions: np.array, truth: np.array, bbox):
    '''
    video_name - video name

    '''
    # Choose video
    if video_path.endswith("video1.avi"):
        first_frame = 10
        last_frame = 42
    elif video_path.endswith("video2.avi"):
        first_frame = 3
        last_frame = 39
    elif video_path.endswith("video3.avi"):
        first_frame = 1
        last_frame = 60

    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(1, first_frame)

    bbox_width = bbox[2]
    bbox_height = bbox[3]

    distances = positions - truth
    distances = np.linalg.norm(distances, axis=1)

    fig, ax = plt.subplots(1)
    ret, frame = vidcap.read()
    im = ax.imshow(frame)

    current_rect = []

    def animate(i):

        if len(current_rect) > 0:
            current_rect[0].remove()
            current_rect[1].remove()
            current_rect.pop(0)
            current_rect.pop(0)

        # Get frame
        vidcap.set(1, i + first_frame)
        ret, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        im.set_data(frame)

        pos_rect = ax.add_patch(Rectangle((positions[i, 0] - 0.5 * bbox_width, positions[i, 1] - 0.5 * bbox_height),
                                bbox_width, bbox_height, fill=False, edgecolor='blue', lw=2, animated=True))

        truth_rect = ax.add_patch(Rectangle((truth[i, 0] - 0.5 * bbox_width, truth[i, 1] - 0.5 * bbox_height),
                                  bbox_width, bbox_height, fill=False, edgecolor='green', lw=1, animated=True))

        ax.set_title(
            f"Distance: {distances[i]:.2f} px, Frame: {i + first_frame}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Distance")

        current_rect.append(pos_rect)
        current_rect.append(truth_rect)

        return pos_rect, truth_rect

    # print("Animating...")
    ani = FuncAnimation(fig, animate, interval=40, blit=True,
                        repeat=True, frames=last_frame - first_frame)

    # print("Saving...")
    ani.save(
        f"{os.path.join(out_dir, 'dist')}.gif", dpi=300, writer=PillowWriter(fps=10))

    plt.close()


if __name__ == "__main__":
    video_name = 'video3.avi'
    data = np.load(f'out/{video_name.split(".")[0]}.npz')

    pre = data["pre"]
    post = data["post"]
    bbox = data["bbox"]

    truth = np.load(f'out/{video_name.split(".")[0]}_truth.npy')

    make_gif(video_name, "out/", positions=pre,
             truth=truth, bbox=bbox, name="pre")
