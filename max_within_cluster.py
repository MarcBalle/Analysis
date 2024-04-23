""" Plot the most dissimilar poses, in terms of L2 distance, inside each cluster """

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from utils.utils import plot_poses


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="directory containing the data")
    parser.add_argument("--output", type=str, required=True, help="output directory")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # TODO: are labels and clusters in the same order? e.g.: label 0 -> centroids[0]
    keypoints = np.load(os.path.join(args.root, "samples.npz"))["arr_0"]
    centroids = np.load(os.path.join(args.root, "cluster_centers.npz"))["arr_0"]
    labels = np.load(os.path.join(args.root, "labels.npz"))["arr_0"]

    clasification = {}
    for label in np.unique(labels):
        clasification[label] = []

    for kpt, label in zip(keypoints, labels):
        clasification[label].append(kpt)

    # Compute (dis)similarity matrix
    pairs = {}
    for k in clasification.keys():
        keypoints = clasification[k]
        N_k = len(keypoints)
        similarity = np.zeros((N_k, N_k))  # this is a symmetric matrix, would be enough to compute upper/lower triangle
        for i in range(N_k):
            k0 = keypoints[i]
            for j in range(N_k):
                k1 = keypoints[j]
                ssd = np.linalg.norm(k0 - k1)  # is this L2 norm??
                similarity[i, j] = ssd

        # Get the pair of poses which differ the most inside of the cluster
        idx = np.argmax(similarity)
        row = idx // N_k
        col = idx % N_k

        pairs[k] = (keypoints[row], keypoints[col])

    for i, k in enumerate(pairs.keys()):
        k0, k1 = pairs[k]
        k0, k1 = k0.reshape((17, 3)), k1.reshape((17, 3))
        fig, ax = plt.subplots(1, 1, figsize=(30, 10), subplot_kw={"projection": "3d"})
        plot_poses(ax, k0)
        plot_poses(ax, k1)
        plt.show()
        plt.savefig(os.path.join(args.output, f"cluster_{i}.png"))
