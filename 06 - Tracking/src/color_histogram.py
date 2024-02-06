from typing import Any
import numpy as np


def color_histogram(
    top_x: int,
    top_y: int,
    bottom_x: int,
    bottom_y: int,
    image: np.array,
    hist_bin: int
) -> np.array:
    """
    Compute color histogram of the image in the given bounding box.
    """
    bounded_img = image[top_y:bottom_y, top_x:bottom_x]
    bounded_img = np.reshape(bounded_img, (-1, bounded_img.shape[-1]))
    hist = np.histogramdd(bounded_img, bins=(hist_bin, hist_bin, hist_bin))
    # return stacked histogram as np.array
    return np.stack(hist[1], axis=-1).flatten()
