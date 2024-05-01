import os

import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform

SKELETON = [
    [0, 1],
    [1, 2],
    [2, 3],
    [0, 4],
    [4, 5],
    [5, 6],
    [0, 7],
    [7, 8],
    [8, 9],
    [9, 10],
    [8, 11],
    [11, 12],
    [12, 13],
    [8, 14],
    [14, 15],
    [15, 16],
]

angle_x = 60 * (np.pi / 180)
angle_y = 180 * (np.pi / 180)
angle_z = -30 * (np.pi / 180)

Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])


def video_to_frames(video, percentage=1.0):
    """Read video into its frames"""
    cap = cv2.VideoCapture(video)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = int(video_length * percentage)
    frame_idxs = np.random.randint(0, video_length - 1, (n,))

    frames = []

    # TODO: this approach might be ways slower than simply while cap.isOpened() and read all
    for f in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        _, img = cap.read()
        frames.append(img)

    cap.release()
    cv2.destroyAllWindows()

    return frames, frame_idxs


def frames_to_labels(video_indices, labels, output_path, percentage=0.8):
    """
    Assign frames from videos to labels

    :param video_indices: association between keypoints array indices and video file
    :type video_indices: dict
    :param labels: labels
    :type labels: list

    """
    labels_per_video = {video_path: labels[video_indices[video_path]] for video_path in video_indices.keys()}

    unique_labels = np.unique(labels)

    for l in unique_labels:
        os.makedirs(os.path.join(output_path, f"{l}"), exist_ok=True)

    for video_path in video_indices.keys():
        frames, frame_idxs = video_to_frames(video_path, percentage)
        video_labels = labels_per_video[video_path][frame_idxs]

        video_name, ext = os.path.splitext(os.path.basename(video_path))
        for i, (frame, label) in enumerate(zip(frames, video_labels)):
            cv2.imwrite(os.path.join(output_path, f"{label}", f"{video_name}_{i:02}.png"), frame)

        del frames


def plot_poses(ax, pose, color=None, c=None, cmap=None, title=None):
    for joint in SKELETON:
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=c, cmap=cmap)
        ax.plot(
            [pose[joint[0], 0], pose[joint[1], 0]],
            [pose[joint[0], 1], pose[joint[1], 1]],
            zs=[pose[joint[0], 2], pose[joint[1], 2]],
            color=color,
        )
        ax.view_init(azim=80)
        ax.set_title(title)


def get_extreme_samples(samples, mode="distant", viz=False):
    """
    Return the most dissimilar samples based on Euclidian distance

    :param samples: all data points
    :param mode: "distant" or "close"
    """
    similarity = squareform(pdist(samples))

    # Avoid taking the diagonal samples
    if mode == "close":
        np.fill_diagonal(similarity, np.inf)
    idx = np.argmax(similarity) if mode == "distant" else np.argmin(similarity)

    N = len(samples)
    row = idx // N
    col = idx % N

    k0, k1 = samples[row], samples[col]
    # If samples are to be visualized, apply reshape and rotation
    if viz:
        k0 = k0.reshape(17, 3)
        k1 = k1.reshape(17, 3)

        k0 -= k0[0]
        k1 -= k1[0]

        k0 = k0 @ Rx.T @ Ry.T @ Rz.T
        k1 = k1 @ Rx.T @ Ry.T @ Rz.T

    return k0, k1


def max_distance_pairs(samples, labels, filter=None, viz=False):
    """
    Return most dissimilar samples of each class based on Euclidian distance

    :param samples: all data points
    :param labels: label per sample
    :param filter: set of labels from where to get max distanced pairs
    """
    clasification = {}
    labels_unique = np.sort(np.unique(labels))
    filter = filter if filter else labels_unique

    clasification = {label: [] for label in filter}

    for kpt, label in zip(samples, labels):
        if label in filter:
            clasification[label].append(kpt)

    # Compute distance matrix
    pairs = {}
    for k in clasification.keys():
        samples = clasification[k]
        k0, k1 = get_extreme_samples(samples, mode="distant", viz=True)
        pairs[k] = (k0, k1)

    return pairs


def variance_per_joint(samples, labels, filter=None):
    """
    Compute variance per joint and per class

    :param samples: all data points
    :param labels: label per sample
    """

    labels_unique = np.sort(np.unique(labels))
    filter = filter if filter else labels_unique

    clasification = {label: [] for label in filter}

    for kpt, label in zip(samples, labels):
        if label in filter:
            clasification[label].append(kpt)

    covariances_per_class = {}
    for k in clasification.keys():
        kpts = clasification[k]
        covs = []
        for i in range(17):
            joint = np.array([kpt.reshape(17, 3)[i] for kpt in kpts])

            # Compute determinant of covariance matrix as a measure of general variance
            cov_det = np.linalg.det(np.cov(joint, rowvar=False))
            covs.append(cov_det)

        covariances_per_class[k] = covs

    return covariances_per_class
