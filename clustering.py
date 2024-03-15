""" Perform clusterin on the 3D poses using classical algorithms """

import os
import glob
import argparse

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="dir where all npz are stored, potentially in subdirs")
    parser.add_argument("--rel_path", type=str, required=True, help="relative common path to find all npz in root dir")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if os.path.isdir(args.root):
        filenames = []
        subdirs = os.listdir(args.root)
        subdirs.sort()
        for subdir in subdirs:
            kpts_files = glob.glob(os.path.join(args.root, subdir, args.rel_path, "3d_*"))
            filenames.extend(kpts_files)

        keypoints = [np.load(filename)["keypoints"].reshape((-1, 17 * 3)) for filename in filenames]

        x = keypoints.pop(0)

        for kpt in keypoints:
            x = np.concatenate((x, kpt), axis=0)

    elif os.path.isfile(args.root):
        x = np.load(args.root)["keypoints"].reshape((-1, 17 * 3))[1000:-1000, ...]

    else:
        raise FileNotFoundError(f"{args.root} does not exist as a directory or file")

    kmeans = KMeans(n_clusters=25).fit(x)

    np.savez(
        "C:\\Users\\marcw\\master_thesis\\forked\\Analysis\\data\\25centers_kmeans.npz",
        centers=kmeans.cluster_centers_,
    )

    np.savez(
        "C:\\Users\\marcw\\master_thesis\\forked\\Analysis\\data\\25centers_kmeans_labels.npz", labels=kmeans.labels_
    )
