import cv2
import os
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Rectangle


def make_gif(video_path: str, out_dir, pre: np.array, post: np.array, truth: np.array, bbox):
    '''
    video_name - video name

    '''

    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(1, 0)
    number_of_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    bbox_width = bbox[2]
    bbox_height = bbox[3]

    distances1 = pre - truth
    distances2 = post - truth
    distances1 = np.linalg.norm(distances1, axis=1)
    distances2 = np.linalg.norm(distances2, axis=1)

    fig, ax = plt.subplots(1)
    ret, frame = vidcap.read()
    im = ax.imshow(frame)

    current_rect = []

    def animate(i):

        if len(current_rect) > 0:
            current_rect[0].remove()
            current_rect[1].remove()
            current_rect[2].remove()
            current_rect.pop(0)
            current_rect.pop(0)
            current_rect.pop(0)

        # Get frame
        vidcap.set(1, i)
        ret, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        im.set_data(frame)

        pre_rect = ax.add_patch(Rectangle((pre[i, 0] - 0.5 * bbox_width, pre[i, 1] - 0.5 * bbox_height),
                                bbox_width, bbox_height, fill=False, edgecolor='blue', lw=2, animated=True))

        post_rect = ax.add_patch(Rectangle((post[i, 0] - 0.5 * bbox_width, post[i, 1] - 0.5 * bbox_height),
                                           bbox_width, bbox_height, fill=False, edgecolor='red', lw=2, animated=True))

        truth_rect = ax.add_patch(Rectangle((truth[i, 0] - 0.5 * bbox_width, truth[i, 1] - 0.5 * bbox_height),
                                  bbox_width, bbox_height, fill=False, edgecolor='green', lw=1, animated=True))

        distance = (distances1[i] + distances2[i]) / 2

        ax.set_title(
            f"Distance: {distance:.2f} px, Frame: {i}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Distance")

        current_rect.append(pre_rect)
        current_rect.append(post_rect)
        current_rect.append(truth_rect)

        return pre_rect, post_rect, truth_rect

    # print("Animating...")
    ani = FuncAnimation(fig, animate, interval=40, blit=True,
                        repeat=True, frames=number_of_frames - 1)

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
