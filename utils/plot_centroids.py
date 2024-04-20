""" Plot the centers resulting from K-Means. """

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers


def plot_poses(axs, poses, skeleton):
    for ax, pose in zip(axs, poses):
        for joint in skeleton:
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2])
            ax.plot(
                [pose[joint[0], 0], pose[joint[1], 0]],
                [pose[joint[0], 1], pose[joint[1], 1]],
                zs=[pose[joint[0], 2], pose[joint[1], 2]],
            )
            ax.view_init(azim=255)
            # ax.set_title(f"Label {idx[i, j]}")


def update_plot(num, axes, lines, poses, skeleton):
    for ax, line_set, pose in zip(axes.flat, lines, poses):
        for line, edge in zip(line_set, skeleton):
            line.set_data([pose[edge[0]][0], pose[edge[1]][0]], [pose[edge[0]][1], pose[edge[1]][1]])
            line.set_3d_properties([pose[edge[0]][2], pose[edge[1]][2]])
        ax.view_init(azim=num)  # Change the viewing angle
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-0.2, 0.4)
    return lines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centroids", type=str, required=True, help="file path to npz containing centers information")
    parser.add_argument("--labels", type=str, required=True, help="class labels assigned by the cluster")
    parser.add_argument("--num_figures", type=int, required=True, help="number of figures to generate")
    parser.add_argument("--save_dir", type=str, default=".", help="output directory path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    centers = np.load(args.centroids)["centers"]
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

    # Set up formatting for the movie files
    Writer = writers["ffmpeg"]
    writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)

    for i in range(args.num_figures):
        fig, axes = plt.subplots(1, centers_per_fig, figsize=(30, 10), subplot_kw={"projection": "3d"})
        centers_fig = centers[idx[i]]

        # Create a static plot
        plot_poses(axes.flat, centers_fig, skeleton)

        # Save the plot
        plt.savefig(os.path.join(args.save_dir, f"{i:02}.png"))

        # Create lines for edges in each subplot
        lines = [[ax.plot([], [], [], markersize=2)[0] for _ in range(len(skeleton))] for ax in axes.flat]

        # Create the animation
        ani = FuncAnimation(
            fig, update_plot, frames=np.arange(0, 360, 2), fargs=(axes, lines, centers_fig, skeleton), interval=50
        )

        # Save the animation as a video file
        ani.save(os.path.join(args.save_dir, f"{i:02}.mp4"), writer=writer)
