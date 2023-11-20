from mean_shift import meanshift_step
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale


scale = 0.5    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape  # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

X = image_lab


def update(i):
    global X
    print(f"Step {i}")
    X = meanshift_step(X)
    scatter.set_array([X[:, 0], X[:, 1], X[:, 2]])


fig, ax = plt.subplots()
# plot 3D scatter of X
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2])


ani = animation.FuncAnimation(fig, update, frames=19, interval=500)
plt.show()
