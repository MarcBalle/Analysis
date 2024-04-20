""" 
Perform clustering on the 3D poses using classical algorithms 

The keypoints should all be in a directory called "keypoints". 

Additionally, all videos corresponding to the the npz files can be saved under "videos" folder 
at the same level as "keypoints folder". If so, all keypoints will be match with their corresponding frames. 

"""

import os
import glob
import argparse

import numpy as np
from sklearn.cluster import KMeans

from utils.utils import frames_to_labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="directory containing the data")
    parser.add_argument("--output", type=str, required=True, help="output directory")
    parser.add_argument(
        "--k",
        type=int,
        required=False,
        default=10,
        help="number of cluster in k-means",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    filenames = glob.glob(os.path.join(args.root, "keypoints", "*.npz"))
    keypoints = []
    keypoints_to_video = {}
    offset = 0
    for fname in filenames:
        # Read keypoints
        kpt = np.load(fname)["keypoints"].reshape((-1, 17 * 3))
        n_frames = kpt.shape[0]
        keypoints.append(kpt)

        # Assign keypoints with video file
        basename, ext = os.path.splitext(os.path.basename(fname))
        video = os.path.join(args.root, "videos", basename + ".mp4")
        keypoints_to_video[video] = range(offset, offset + n_frames)

        offset = offset + n_frames

    x = keypoints.pop(0)

    for kpt in keypoints:
        x = np.concatenate((x, kpt), axis=0)

    del keypoints

    n_centroids = np.random.randint(25, 100, (10,))  # random search
    for k in n_centroids:
        os.makedirs(os.path.join(args.output, f"{k}"), exist_ok=True)

        kmeans = KMeans(n_clusters=args.k).fit(x)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        frames_to_labels(keypoints_to_video, labels=labels, output_path=args.output, k=k)

        np.savez(os.path.join(args.output, "samples"), x)
        np.savez(os.path.join(args.output, "cluster_centers"), centroids)
        np.savez(os.path.join(args.output, "labels"), labels)
