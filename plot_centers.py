""" Plot the centers resulting from the clustering. """

import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centers", type=str, required=True, help="file path to npz containing centers information")
    parser.add_argument("--labels", type=str, required=True, help="class labels assigned by the cluster")
    parser.add_argument("--num_figures", type=int, required=True, help="number of figures to generate")
    parser.add_argument("--match_images", action="store_true", help="whether matching images to cluster labels")
    parser.add_argument("--video", type=str, help="video path")
    parser.add_argument(
        "--skipped_frames",
        type=int,
        default=0,
        help="number of cropped frames (beginning and end) by the time of clustering",
    )
    parser.add_argument("--save_dir", type=str, default=".", help="output directory path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    centers = np.load(args.centers)["centers"]
    centers = centers.reshape((-1, 17, 3))

    n_centers = centers.shape[0]
    assert n_centers % args.num_figures == 0, "Number of figures must be multiple of the number of cluster centers"
    centers_per_fig = int(n_centers / args.num_figures)

    angle_x = 60 * (np.pi / 180)
    angle_y = 180 * (np.pi / 180)
    angle_z = -30 * (np.pi / 180)

    rot_x = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    rot_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
    rot_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])

    # Hip joint is center of coordinates
    centers -= np.expand_dims(centers[:, 0, :], axis=1)

    centers = centers @ rot_x.T @ rot_y.T @ rot_z.T

    skeleton = [
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

    idx = np.array(list(range(0, n_centers))).reshape((args.num_figures, centers_per_fig))

    for i in range(args.num_figures):
        fig = plt.figure(figsize=(30, 10))
        for j in range(centers_per_fig):
            center = centers[idx[i, j]]
            ax = fig.add_subplot(1, centers_per_fig, j + 1, projection="3d")
            for joint in skeleton:
                ax.scatter(center[:, 0], center[:, 1], center[:, 2])
                ax.plot(
                    [center[joint[0], 0], center[joint[1], 0]],
                    [center[joint[0], 1], center[joint[1], 1]],
                    zs=[center[joint[0], 2], center[joint[1], 2]],
                )
                ax.view_init(azim=225)
                ax.set_title(f"Label {idx[i, j]}")
        # plt.savefig(os.path.join(args.save_dir, f"{i}.png"))

    if args.match_images:
        labels = np.load(args.labels)["labels"]
        unique_labels = np.unique(labels)

        os.makedirs(os.path.join(args.save_dir, "frames"), exist_ok=True)

        for l in unique_labels:
            os.makedirs(os.path.join(args.save_dir, "frames", f"{l}"), exist_ok=True)

        cap = cv2.VideoCapture(args.video)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = [cap.read()[1] for _ in range(video_length)]
        frames = frames[args.skipped_frames : -args.skipped_frames]  # skip cropped frames while clustering

        assert (
            len(frames) == labels.shape[0]
        ), f"Number of frames {len(frames)} and corresponding labels {labels.shape[0]} do not match in number"

        for i, f in enumerate(frames):
            f_label = labels[i]
            cv2.imwrite(os.path.join(args.save_dir, "frames", f"{f_label}", f"frame_{i}.png"), f)
