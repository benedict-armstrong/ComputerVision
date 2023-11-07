import numpy as np


def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please

    distances = np.zeros((desc1.shape[0], desc2.shape[0]))
    for i in range(desc1.shape[0]):
        for j in range(desc2.shape[0]):
            distances[i][j] = np.sum((desc1[i] - desc2[j])**2)
    return distances
    # raise NotImplementedError


def match_descriptors(desc1, desc2, method="one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m, 2) numpy array storing the indices of the matches
    '''

    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way":  # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        matches = np.argmin(distances, axis=1)
        # add second row to each match
        matches = np.vstack((np.arange(q1), matches)).T

        # raise NotImplementedError
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        matches_1 = np.argmin(distances, axis=1)
        matches_2 = np.argmin(distances, axis=0)

        matches = []
        for i in range(q1):
            if matches_2[matches_1[i]] == i:
                matches.append([i, matches_1[i]])

        matches = np.array(matches)
        # raise NotImplementedError
    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        # print(distances)

        two_closest = np.partition(distances, 2, axis=1)[:, 0:2]
        ratio = two_closest[:, 0] / two_closest[:, 1]

        matches = np.argmin(distances, axis=1) * (ratio < ratio_thresh)
        matches = np.vstack((np.arange(q1), matches)).T
        matches = matches[matches[:, 1] != 0]
        # raise NotImplementedError
    else:
        raise NotImplementedError
    return matches
