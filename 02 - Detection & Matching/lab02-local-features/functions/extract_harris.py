import cv2
import numpy as np

from scipy import signal  # for the scipy.signal.convolve2d function
from scipy import ndimage  # for the scipy.ndimage.maximum_filter

# Harris corner detector


def extract_harris(img, sigma=1.0, k=0.05, thresh=1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.

    kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    I_x = signal.convolve2d(img, np.transpose(kernel), mode='same')
    I_y = signal.convolve2d(img, kernel, mode='same')

    cv2.imwrite("I_x.png", I_x * 255)
    cv2.imwrite("I_y.png", I_y * 255)

    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    I_xx = cv2.GaussianBlur(I_x ** 2, (5, 5), sigma,
                            borderType=cv2.BORDER_REPLICATE)
    I_xy = cv2.GaussianBlur(I_x * I_y, (5, 5), sigma,
                            borderType=cv2.BORDER_REPLICATE)
    I_yy = cv2.GaussianBlur(I_y ** 2, (5, 5), sigma,
                            borderType=cv2.BORDER_REPLICATE)

    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    M = np.zeros((*img.shape, 2, 2))
    M[:, :, 0, 0] = I_xx
    M[:, :, 1, 1] = I_yy
    M[:, :, 0, 1] = I_xy
    M[:, :, 1, 0] = I_xy

    # cv2.imwrite("I_xx.png", I_xx * 1e3)
    # cv2.imwrite("I_xy.png", I_xy * 1e3)
    # cv2.imwrite("I_yy.png", I_yy * 1e3)

    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    C = np.zeros(img.shape)
    C = np.linalg.det(M) - k * np.square(np.trace(M, axis1=2, axis2=3))

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format

    # find max in 3x3 neighborhood setting all other values in neighborhood to 0
    C = C * (C == ndimage.maximum_filter(C, size=3))
    corners = np.argwhere(C > thresh)

    # plot keypoints on the image
    # for x, y in corners:
    #     cv2.circle(img, (y, x), 1, (0, 0, 255), -1)

    # cv2.imwrite(f"out.png", img * 255.0)

    # switch x and y
    corners[:, [1, 0]] = corners[:, [0, 1]]

    return corners, C
