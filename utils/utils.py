import os

import cv2
import numpy as np

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
    labels_per_video = {}
    for video_path in video_indices.keys():
        labels_per_video[video_path] = labels[video_indices[video_path]]

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


def plot_poses(ax, pose):
    for joint in SKELETON:
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2])
        ax.plot(
            [pose[joint[0], 0], pose[joint[1], 0]],
            [pose[joint[0], 1], pose[joint[1], 1]],
            zs=[pose[joint[0], 2], pose[joint[1], 2]],
        )
        ax.view_init(azim=255)
        # ax.set_title(f"Label {idx[i, j]}")
