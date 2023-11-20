import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale


def distance(x: np.array, X: np.array) -> np.array:
    """
    Compute the distance between a given point and all other points that are within a specific radius.

    radius = +∞
    """
    return np.sqrt(np.sum((x - X) ** 2, axis=1))


def gaussian(dist: np.array, bandwidth: float) -> np.array:
    """
    Compute weight based on distance
    """

    return np.exp(-0.5 * (dist / bandwidth) ** 2)


def update_point(weight: np.array, X: np.array) -> np.array:
    """
    Update point position based on weight
    """
    return np.sum(weight.reshape(-1, 1) * X, axis=0) / np.sum(weight)


def meanshift_step(X: np.array, bandwidth=2) -> np.array:
    """
    Run one step of mean-shift algorithm on all points using np
    """
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X[i] = update_point(weight, X)
    return X


def meanshift(X: np.array):
    for i in range(1):
        print(f"Step {i}")
        X = meanshift_step(X)
    return X


if __name__ == '__main__':
    scale = 0.5    # downscale the image to run faster

    # Load image and convert it to CIELAB space
    image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
    image_lab = color.rgb2lab(image)
    shape = image_lab.shape  # record image shape
    image_lab = image_lab.reshape([-1, 3])  # flatten the image

    # Run your mean-shift algorithm
    t = time.time()
    X = meanshift(image_lab)
    t = time.time() - t
    print('Elapsed time for mean-shift: {}'.format(t))

    # Load label colors and draw labels as an image
    colors = np.load('colors.npz')['colors']
    colors[colors > 1.0] = 1
    colors[colors < 0.0] = 0

    centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

    # print(pd.DataFrame(labels))

    # convert result back to RGB
    out = color.lab2rgb(image_lab[labels].reshape(-1, 1, 3))

    result_image = out.reshape(shape)

    # resize result image to original resolution
    result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)
    result_image = (result_image * 255).astype(np.uint8)
    io.imsave('result.png', result_image)
