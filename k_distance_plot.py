""" 
A k-dtance plot shows the average distance of each point to its k-th NN. 
It is used to choose the epsilon value for DBSCAN.
"""

import argparse
import os
import glob

from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kpts", type=str, required=True, help="keypoints directory")

    args = parser.parse_args()

    filenames = glob.glob(os.path.join(args.kpts, "*.npz"))
    keypoints = []
    for fname in filenames:
        # Read keypoints
        kpt = np.load(fname)["keypoints"].reshape((-1, 17 * 3))
        keypoints.append(kpt)

    x = keypoints.pop(0)

    for kpt in keypoints:
        x = np.concatenate((x, kpt), axis=0)

    # n_neighbours is set to minPts in DBSCAN, which is suggested to be 2 * n_dims
    neighbors = NearestNeighbors(n_neighbors=2 * 17 * 3)
    neighbors_fit = neighbors.fit(x)
    distances, indices = neighbors_fit.kneighbors(x)

    distances = np.mean(distances[:, 1:], axis=1)
    distances = np.sort(distances)
    plt.plot(distances)
    plt.savefig("C:\\Users\\marcw\\master_thesis\\forked\\Analysis\\data\\kdistances.png")
