""" Perform clusterin on the 3D poses using classical algorithms """

import os
import glob
import argparse

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, required=True, help="dir where all npz are stored, potentially in subdirs"
    )
    parser.add_argument("--rel_path", type=str, required=True, help="relative common path to find all npz in root dir")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert os.path.isdir(args.root_dir), f"{args.keypoints} is not a directory"

    filenames = []
    subdirs = os.listdir(args.root_dir)
    subdirs.sort()
    for subdir in subdirs:
        kpts_files = glob.glob(os.path.join(args.root_dir, subdir, args.rel_path, "3d_*"))
        filenames.extend(kpts_files)

    keypoints = [np.load(filename)["keypoints"].reshape((-1, 17 * 3)) for filename in filenames]

    x = keypoints.pop(0)

    for kpt in keypoints:
        x = np.concatenate((x, kpt), axis=0)

    kmeans = KMeans(n_clusters=100).fit(x)
    np.savez("C:\\Users\\marcw\\master_thesis\\100centers_kmeans.npz", centers=kmeans.cluster_centers_)
