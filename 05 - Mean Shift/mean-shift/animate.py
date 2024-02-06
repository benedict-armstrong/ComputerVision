import os
from mean_shift import meanshift_step
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale


scale = 0.5    # downscale the image to run faster
image_path = "eth.jpg"
image_name, _ = os.path.splitext(image_path)
steps = 15
bandwidth = 5
with_color = True
distance = False


# Load image and convert it to CIELAB space
image = rescale(io.imread(image_path), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape  # record image shape
if distance:
    image_lab = np.concatenate(
        (image_lab, np.indices(shape[:2]).transpose(1, 2, 0)),
        axis=2
    )

    image_lab[:, :, 3:] *= 0.2

    # flatten the image with 5 channels
    image_lab = image_lab.reshape([-1, 5])
else:
    image_lab = image_lab.reshape([-1, 3])  # flatten the image

X = image_lab


def update(i):
    global X
    global scatter
    print(f"animation step {i}")
    cache_file = f".cache/{image_name}/ms_{'location_0.2_' if distance else ''}{bandwidth}_{i}.npz"
    if os.path.exists(cache_file):
        print(f"loading from {cache_file}")
        X = np.load(cache_file)['X']
    else:
        print(f"running meanshift_step {i}")
        X = meanshift_step(X, bandwidth=bandwidth)

    # update scatter convert CIELAB color to RGB color
    scatter.set_offsets(X[:, :2])
    scatter.set_3d_properties(X[:, 2], 'z')
    if with_color:
        scatter.set_color(color.lab2rgb(X[:, :3]))

    # if i == steps - 1:
    #     plt.savefig("last.png")


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.title.set_text(
    f"bandwidth={bandwidth}, steps={steps}{', with physical distance' if distance else ''}")
ax.set_xlabel('L')
ax.set_ylabel('a')
ax.set_zlabel('b')
# ax.set_xlim(0, 100)
# ax.set_ylim(-128, 127)
# ax.set_zlim(-128, 127)
# plot 3D scatter of X
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2])
scatter.set_color(color.lab2rgb(X[:, :3]))

# plt.savefig(f"first.png")

if distance:
    # set color based on position in RGB color space r,g,b -> [0, 1]
    c = X[:, 3:] * 50

    # add a column of zeros to make it 3D
    c = np.concatenate((c, np.ones((c.shape[0], 1)) * 0.5), axis=1)

    # make sure all values are in [0, 1] range by scaling down (for both axis independently)
    c[:, 0] /= np.max(c[:, 0])
    c[:, 1] /= np.max(c[:, 1])

    scatter.set_color(c)

ani = animation.FuncAnimation(fig, update, frames=steps)

ani.save(f"images/{image_name}/{'location_' if distance else ''}{'color_' if with_color else''}{bandwidth}_{steps}.gif",
         writer='imagemagick', fps=3)
plt.show()
