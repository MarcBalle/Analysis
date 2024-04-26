""" 
Perform clustering on the 3D poses using classical algorithms 

The keypoints should all be in a directory called "keypoints". 

Additionally, all videos corresponding to the the npz files can be saved under "videos" folder 
at the same level as "keypoints folder". If so, all keypoints will be match with their corresponding frames. 

"""

import os
import glob
import argparse
import pickle

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from utils.utils import frames_to_labels


class Kmeans:
    def __init__(self, output_dir, keypoints_to_video, percentage=0.2, save_frames=False) -> None:
        # Random search for number of clusters
        self.k = np.random.randint(25, 100, (10,))

        self.output_dir = output_dir

        self.keypoints_to_video = keypoints_to_video
        self.percentage = percentage

        self.save_frames = save_frames

    def run(self, x):
        for k in self.k:
            out_subdir = os.path.join(self.output_dir, f"{k}")
            os.makedirs(out_subdir, exist_ok=True)

            kmeans = KMeans(n_clusters=k).fit(x)
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_

            if self.save_frames:
                frames_to_labels(
                    self.keypoints_to_video, labels=labels, output_path=out_subdir, percentage=self.percentage
                )

            np.savez(os.path.join(out_subdir, "cluster_centers"), centroids)
            np.savez(os.path.join(out_subdir, "labels"), labels)


class GMM:
    def __init__(self, output_dir, keypoints_to_video, percentage=0.2, save_frames=False) -> None:
        # Random search for number of components
        self.k = np.random.randint(25, 100, (10,))

        self.output_dir = output_dir

        self.keypoints_to_video = keypoints_to_video
        self.percentage = percentage

        self.save_frames = save_frames

    def run(self, x):
        for k in self.k:
            out_subdir = os.path.join(self.output_dir, f"{k}")
            os.makedirs(out_subdir, exist_ok=True)

            gmm = GaussianMixture(n_components=k).fit(x)
            means = gmm.means_
            labels = gmm.predict(x)

            if self.save_frames:
                frames_to_labels(
                    self.keypoints_to_video, labels=labels, output_path=out_subdir, percentage=self.percentage
                )

            np.savez(os.path.join(out_subdir, "means"), means)
            np.savez(os.path.join(out_subdir, "labels"), labels)


class DBSCAN_:
    def __init__(
        self, min_eps, max_eps, min_samples, output_dir, keypoints_to_video, percentage=0.2, save_frames=False
    ) -> None:
        # Random search for epsilon
        self.eps = np.random.uniform(min_eps, max_eps, (10,))
        self.min_samples = min_samples

        self.output_dir = output_dir

        self.keypoints_to_video = keypoints_to_video
        self.percentage = percentage

        self.save_frames = save_frames

    def run(self, x):
        for eps in self.eps:
            out_subdir = os.path.join(self.output_dir, f"{eps}")
            os.makedirs(out_subdir, exist_ok=True)

            dbscan = DBSCAN(eps=eps, min_samples=self.min_samples).fit(x)
            labels = dbscan.labels_

            if self.save_frames:
                frames_to_labels(
                    self.keypoints_to_video, labels=labels, output_path=out_subdir, percentage=self.percentage
                )

            np.savez(os.path.join(out_subdir, "labels"), labels)


class HDBSCAN_:
    def __init__(
        self, min_size, max_size, output_dir, keypoints_to_video, percentage=0.2, save_frames=False, min_samples=None
    ) -> None:
        # Random search for minimum cluster size
        self.cluster_sizes = np.random.randint(min_size, max_size, (10,))
        self.min_samples = min_samples

        self.output_dir = output_dir

        self.keypoints_to_video = keypoints_to_video
        self.percentage = percentage

        self.save_frames = save_frames

    def run(self, x):
        for cluster_size in self.cluster_sizes:
            out_subdir = os.path.join(self.output_dir, f"{cluster_size}")
            os.makedirs(out_subdir, exist_ok=True)

            hdbscan = HDBSCAN(min_cluster_size=cluster_size, min_samples=self.min_samples).fit(x)
            medoids = hdbscan.medoids_
            labels = hdbscan.labels_

            if self.save_frames:
                frames_to_labels(
                    self.keypoints_to_video, labels=labels, output_path=out_subdir, percentage=self.percentage
                )

            np.savez(os.path.join(out_subdir, "medoids"), medoids)
            np.savez(os.path.join(out_subdir, "labels"), labels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="directory containing the data")
    parser.add_argument("--output", type=str, required=True, help="output directory")
    parser.add_argument(
        "--method",
        type=str,
        choices=["kmeans", "gmm", "dbscan", "hdbscan"],
        default="kmeans",
        help="clustering method to apply form scikit-learn",
    )
    # TODO: add arguments specific for each clustering methods e.g.: k for Kmeans or eps for DBSCAN

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
        # video = os.path.join(args.root, "videos", basename + ".mp4")
        video = os.path.join("Z:\\workfiles\\ballesanchezmarc\\data\\driveact", "videos", basename + ".mp4")
        keypoints_to_video[video] = range(offset, offset + n_frames)

        offset = offset + n_frames

    x = keypoints.pop(0)

    for kpt in keypoints:
        x = np.concatenate((x, kpt), axis=0)

    del keypoints

    if args.method == "kmeans":
        output_dir = os.path.join(args.output, "kmeans")
        clustering = Kmeans(output_dir, keypoints_to_video)
    elif args.method == "gmm":
        output_dir = os.path.join(args.output, "gmm")
        clustering = GMM(output_dir, keypoints_to_video)
    elif args.method == "dbscan":
        output_dir = os.path.join(args.output, "dbscan")
        clustering = DBSCAN_(
            min_eps=0.15,
            max_eps=0.25,
            min_samples=2 * 17 * 3,  # 2 * n_dim
            output_dir=output_dir,
            keypoints_to_video=keypoints_to_video,
        )
    elif args.method == "hdbscan":
        output_dir = os.path.join(args.output, "hdbscan")
        clustering = HDBSCAN_(min_size=700, max_size=8000, output_dir=output_dir, keypoints_to_video=keypoints_to_video)
    else:
        raise NotImplementedError(f"Clustering method {args.method} is not implemented")

    clustering.run(x)

    np.savez(os.path.join(output_dir, "samples"), x)
    with open(os.path.join(output_dir, "keypoints_to_video.pkl"), "wb") as handle:
        pickle.dump(keypoints_to_video, handle)
