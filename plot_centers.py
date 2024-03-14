""" Plot the centers resulting from the clustering. """

import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centers", type=str, required=True, help="file path to npz containing centers information")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    centers = np.load(args.centers)["centers"]
    centers = centers.reshape((-1, 17, 3))

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

    idx = np.array(list(range(0, 100))).reshape((10, 10))

    for i in range(10):
        fig = plt.figure(figsize=(30, 10))
        for j in range(10):
            center = centers[idx[i, j]]
            ax = fig.add_subplot(2, 5, j + 1, projection="3d")
            for joint in skeleton:
                ax.scatter(center[:, 0], center[:, 1], center[:, 2])
                ax.plot(
                    [center[joint[0], 0], center[joint[1], 0]],
                    [center[joint[0], 1], center[joint[1], 1]],
                    zs=[center[joint[0], 2], center[joint[1], 2]],
                )
        plt.savefig(f"C:\\Users\\marcw\\master_thesis\\{i}.png")
