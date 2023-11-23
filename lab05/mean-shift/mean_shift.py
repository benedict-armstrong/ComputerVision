import os
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
    # return np.sqrt(np.sum((x - X) ** 2, axis=1))
    return np.linalg.norm(x - X, axis=1)


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
    for _ in range(steps):
        X = meanshift_step(X, bandwidth=bandwidth)
    return X


def test_meanshift(image_path: str, bandwidth=5.0, steps=10, scale=0.5, location_penalty=None):

    image_name, _ = os.path.splitext(image_path)
    if not os.path.isdir(f'images/{image_name}'):
        os.makedirs(f'images/{image_name}')
    if not os.path.isdir(f'.cache/{image_name}'):
        os.makedirs(f'.cache/{image_name}')

    # Load image and convert it to CIELAB space
    original_image = io.imread(image_path)
    image = rescale(original_image, scale, channel_axis=-1)
    image_lab = color.rgb2lab(image)
    shape = image_lab.shape  # record image shape

    if location_penalty:
        image_lab = np.concatenate(
            (image_lab, np.indices(shape[:2]).transpose(1, 2, 0)),
            axis=2
        )

        image_lab[:, :, 3:] *= location_penalty

        # flatten the image with 5 channels
        image_lab = image_lab.reshape([-1, 5])
    else:
        # flatten the image with 3 channels
        image_lab = image_lab.reshape([-1, 3])

    X = image_lab
    for i in range(steps):

        name = f"{f'location_{location_penalty}_' if location_penalty else '' }{bandwidth}_{i}"

        # Run your mean-shift algorithm
        t = time.time()
        cache_file = f'.cache/{image_name}/ms_{name}.npz'
        if os.path.exists(cache_file):
            data = np.load(cache_file)
            X = data['X']
        else:
            X = meanshift_step(X, bandwidth=bandwidth)
            np.savez(cache_file, X=X)

        t = time.time() - t
        print('Elapsed time for mean-shift: {}'.format(t))

        # Load label colors and draw labels as an image
        colors = np.load('colors.npz')['colors']
        colors_2 = np.load('colors_2.npz')['colors']
        colors[colors > 1.0] = 1
        colors[colors < 0.0] = 0

        centroids, labels = np.unique(
            (X / 4).round(), return_inverse=True, axis=0)

        print(
            f"\tFound: {len(centroids)} centroids in {i} steps with bandwidth {bandwidth}")

        if len(centroids) > 500:
            print("\t\tToo many centroids, skipping")
            continue
        if len(centroids) > 23:
            result_image = colors_2[labels].reshape(shape)
        else:
            result_image = colors[labels].reshape(shape)

        # resize result image to original resolution
        result_image = rescale(result_image, 1 / scale,
                               order=0, channel_axis=-1)
        result_image = (result_image * 255).astype(np.uint8)
        io.imsave(f'images/{image_name}/result_{name}.png', result_image)

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
            f'images/{image_name}/outlined_{name}.png', result_image)

        # save image in using centroids as colors
        temp = centroids[labels][:, :3]
        result_image = color.lab2rgb(temp)
        result_image *= 4
        result_image[result_image > 1.0] = 1
        result_image = result_image.reshape(shape)
        result_image = rescale(result_image, 1 / scale,
                               order=0, channel_axis=-1)
        result_image = (result_image * 255).astype(np.uint8)
        io.imsave(
            f'images/{image_name}/result_{name}_{bandwidth}_{i}_centroids.png', result_image)

    # save final result as result.png with colors from image
    # temp = centroids[labels][:, :3]
    # result_image = color.lab2rgb(temp)
    # result_image *= 4
    # result_image[result_image > 1.0] = 1
    # result_image = result_image.reshape(shape)
    # result_image = rescale(result_image, 1 / scale,
    #                        order=0, channel_axis=-1)
    # result_image = (result_image * 255).astype(np.uint8)
    # io.imsave(f'result_{image_name}_{bandwidth}.png', result_image)


if __name__ == '__main__':
    steps = 20
    bandwidth = [4.5]

    # run for all combinations of steps and bandwidth
    for b in bandwidth:
        print(
            f'Running mean-shift with bandwidth={b}, steps={steps}')
        test_meanshift(
            "eth.jpg",
            bandwidth=b,
            steps=steps,
            scale=0.5,
            # location_penalty=0.2
        )
