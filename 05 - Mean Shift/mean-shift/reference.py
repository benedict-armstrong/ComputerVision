import time
import os
import numpy as np
from sklearn.cluster import MeanShift

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale


if __name__ == '__main__':
    scale = 0.5    # downscale the image to run faster

    # Load image and convert it to CIELAB space
    image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
    image_lab = color.rgb2lab(image)
    shape = image_lab.shape  # record image shape

    # append the position of each pixel to its color
    image_lab = np.concatenate(
        (image_lab, np.indices(shape[:2]).transpose(1, 2, 0)),
        axis=2
    )

    # multiply the position with a factor to change its importance
    local_penalty = 0.3
    bandwidth = 10
    image_lab[:, :, 3:] *= local_penalty

    image_lab = image_lab.reshape([-1, 5])  # flatten the image

    # Run your mean-shift algorithm
    t = time.time()

    if os.path.exists(f'.cache/mean-shift_{bandwidth}.npz') and False:
        data = np.load(f'.cache/mean-shift_{bandwidth}.npz')
        labels = data['labels']
        cluster_centers = data['cluster_centers']
    else:
        ms = MeanShift(bandwidth=bandwidth).fit(image_lab)

        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        # np.savez(f'.cache/mean-shift_{bandwidth}.npz', labels=labels,
        #          cluster_centers=cluster_centers)

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    # map each label to color
    label_to_color = {}
    for label in labels_unique:
        label_to_color[label] = np.random.randint(0, 255, 3)

    # map each pixel to its corresponding color
    image_clustered = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            image_clustered[i, j] = label_to_color[labels[i * shape[1] + j]]

    print('time elapsed: %f' % (time.time() - t))

    # save the image
    result_image = rescale(image_clustered, 1 / scale,
                           order=0, channel_axis=-1)
    result_image = (result_image * 255).astype(np.uint8)
    io.imsave(f'reference_{bandwidth}_{local_penalty}.jpg', result_image)

    # save image with outlines drawn over the original image
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
        f'reference_{bandwidth}_{local_penalty}_outlined.jpg', result_image)
