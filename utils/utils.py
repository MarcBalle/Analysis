import os

import cv2
import numpy as np


def video_to_frames(video):
    """Read video into its frames"""
    cap = cv2.VideoCapture(video)

    frames = []
    while cap.isOpened():
        _, img = cap.read()
        frames.append(img)

    cap.release()
    cv2.destroyAllWindows()

    return frames


def frames_to_labels(video_indices, labels, output_path):
    """
    Assign frames from videos to K-Means labels

    :param video_indices: association between keypoints indices and video
    :type video_indices: dict
    :param labels: labels attribute from the KMeans class from sklearn
    :type labels: list

    """
    labels_per_video = {}
    for video_path in video_indices.keys():
        labels_per_video[video_path] = labels[video_indices[video_path]]

    unique_labels = np.unique(labels)

    os.makedirs(os.path.join(output_path, "frames"), exist_ok=True)
    for l in unique_labels:
        os.makedirs(os.path.join(output_path, "frames", f"{l}"), exist_ok=True)

    for video_path in video_indices.keys():
        frames = video_to_frames(video_path)
        video_labels = labels_per_video[video_path]

        video_name = os.path.splitext(os.path.basename(video_path))
        for i, (frame, label) in enumerate(zip(frames, video_labels)):
            cv2.imwrite(os.path.join(output_path, "frames", f"{label}", f"{video_name}_{i:02}.png"), frame)
