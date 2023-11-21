import os
import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

# label colors for visualization (500 random colors)
# colors = np.random.rand(500, 3)
# np.savez('colors_2.npz', colors=colors)
# pass


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


def meanshift(X: np.array, bandwidth=2, steps=5) -> np.array:
    for i in range(steps):
        if os.path.exists(f".cache/ms_{i}_{bandwidth}.npz"):
            data = np.load(f".cache/ms_{i}_{bandwidth}.npz")
            X = data['X']
        else:
            X = meanshift_step(X, bandwidth=bandwidth)
    return X


def test_meanshift(bandwidth=5, steps=10, scale=0.8):

    # Load image and convert it to CIELAB space
    original_image = io.imread('eth.jpg')
    image = rescale(original_image, scale, channel_axis=-1)
    image_lab = color.rgb2lab(image)
    shape = image_lab.shape  # record image shape
    image_lab = image_lab.reshape([-1, 3])  # flatten the image

    # Run your mean-shift algorithm
    t = time.time()
    cache_file = f'.cache/ms_{steps}_{bandwidth}.npz'
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        X = data['X']
    else:
        X = meanshift(image_lab, bandwidth=bandwidth, steps=steps)
        np.savez(cache_file, X=X)
    t = time.time() - t
    print('Elapsed time for mean-shift: {}'.format(t))

    # Load label colors and draw labels as an image
    colors = np.load('colors.npz')['colors']
    colors_2 = np.load('colors_2.npz')['colors']
    colors[colors > 1.0] = 1
    colors[colors < 0.0] = 0

    centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

    print(
        f"\tFound: {len(centroids)} centroids in {steps} steps with bandwidth {bandwidth}")

    if len(centroids) > 500:
        print("\t\tToo many centroids, skipping")
        return
    if len(centroids) > 23:
        result_image = colors_2[labels].reshape(shape)
    else:
        result_image = colors[labels].reshape(shape)

    # resize result image to original resolution
    result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)
    result_image = (result_image * 255).astype(np.uint8)
    io.imsave(f'images/result_{bandwidth}_{steps}.png', result_image)

    # draw outline of segmentation result on original image (with original resolution)
    image_outlined = np.copy(image)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i > 0 and labels[i * shape[1] + j] != labels[(i - 1) * shape[1] + j]:
                image_outlined[i, j] = [255, 0, 0]
            elif j > 0 and labels[i * shape[1] + j] != labels[i * shape[1] + j - 1]:
                image_outlined[i, j] = [255, 0, 0]
    result_image = rescale(image_outlined, 1 / scale,
                           order=0, channel_axis=-1)
    result_image = (result_image * 255).astype(np.uint8)
    io.imsave(
        f'images/outlined_{bandwidth}_{steps}.png', result_image)


if __name__ == '__main__':
    steps = range(1, 15, 1)
    bandwidth = [4]

    # run for all combinations of steps and bandwidth
    for s in steps:
        for b in bandwidth:
            print(
                f'Running mean-shift with bandwidth={b}, steps={s}')
            test_meanshift(bandwidth=b, steps=s)
