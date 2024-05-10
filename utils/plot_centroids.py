""" Plot the centers resulting from K-Means. """

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D

from utils import plot_poses, max_distance_pairs, variance_per_joint, get_extreme_samples, SKELETON, Rx, Ry, Rz


def update_plot(num, axes, lines, centroids, pairs, variances, skeleton):
    lines_cen = lines[:5]
    lines_pairs = lines[5:10]
    # lines_variances = lines[10:]

    axes_cen = axes.flat[:5]
    axes_pairs = axes.flat[5:10]
    # axes_variances = axes.flat[10:]

    for ax, line_set, centroid in zip(axes_cen, lines_cen, centroids):
        for line, edge in zip(line_set, skeleton):
            line.set_data([centroid[edge[0]][0], centroid[edge[1]][0]], [centroid[edge[0]][1], centroid[edge[1]][1]])
            line.set_3d_properties([centroid[edge[0]][2], centroid[edge[1]][2]])
        ax.view_init(azim=num)  # Change the viewing angle

    for ax, line_set, pair in zip(axes_pairs, lines_pairs, pairs.values()):
        k0, k1 = pair
        lines0, lines1 = line_set[: len(skeleton)], line_set[len(skeleton) :]
        for l0, l1, edge in zip(lines0, lines1, skeleton):
            for edge in skeleton:
                l0.set_data([k0[edge[0]][0], k0[edge[1]][0]], [k0[edge[0]][1], k0[edge[1]][1]])
                l0.set_3d_properties([k0[edge[0]][2], k0[edge[1]][2]])
                l0.set_color("r")

                l1.set_data([k1[edge[0]][0], k1[edge[1]][0]], [k1[edge[0]][1], k1[edge[1]][1]])
                l1.set_3d_properties([k1[edge[0]][2], k1[edge[1]][2]])
                l1.set_color("b")
        ax.view_init(azim=num)

    # for ax, line_set, centroid in zip(axes_cen, lines_cen, centroids):
    #     for line, edge in zip(line_set, skeleton):
    #         line.set_data([centroid[edge[0]][0], centroid[edge[1]][0]], [centroid[edge[0]][1], centroid[edge[1]][1]])
    #         line.set_3d_properties([centroid[edge[0]][2], centroid[edge[1]][2]])
    #     ax.view_init(azim=num)  # Change the viewing angle
    #     ax.set_xlim(-0.8, 0.8)
    #     ax.set_ylim(-0.8, 0.8)
    #     ax.set_zlim(-0.2, 0.4)

    return lines_centroids + lines_pairs  # + lines_variances


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centroids", type=str, required=True, help="file path to npz containing centers information")
    parser.add_argument("--kpts", type=str, required=False, help="keypoints file")
    parser.add_argument("--labels", type=str, required=False, help="labels file")
    parser.add_argument("--num_figures", type=int, required=True, help="number of figures to generate")
    parser.add_argument("--centers_fig", type=int, default=5, help="center per figure")
    parser.add_argument("--save_dir", type=str, default=".", help="output directory path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    centers = np.load(args.centroids)["arr_0"].reshape((-1, 17, 3))
    if args.kpts:
        keypoints = np.load(args.kpts)["arr_0"]
    if args.labels:
        labels = np.load(args.labels)["arr_0"]

    n_centers = centers.shape[0]
    assert n_centers <= args.num_figures * args.centers_fig, "Not all center are going to be displayed"

    # Get most distant and closest centroids
    d0, d1 = get_extreme_samples(centers.reshape((-1, 17 * 3)), mode="distant", viz=True)
    c0, c1 = get_extreme_samples(centers.reshape((-1, 17 * 3)), mode="close", viz=True)

    # Plot the most distant and closest centroids
    fig, axes = plt.subplots(1, 2, figsize=(30, 10), subplot_kw={"projection": "3d"})
    plot_poses(axes[0], d0, color="y", title="Most distant centroids")
    plot_poses(axes[0], d1, color="g", title="Most distant centroids")
    plot_poses(axes[1], c0, color="y", title="Closest centroids")
    plot_poses(axes[1], c1, color="g", title="Closest centroids")
    plt.savefig(os.path.join(args.save_dir, "extreme.png"))
    plt.close()

    # Pad the center with 0s to fill the last figure
    if n_centers < args.num_figures * args.centers_fig:
        diff = args.num_figures * args.centers_fig - n_centers
        centers = np.pad(centers, ((0, diff), (0, 0), (0, 0)))
        n_centers = centers.shape[0]

    # Hip joint is center of coordinates
    centers -= np.expand_dims(centers[:, 0, :], axis=1)

    centers = centers @ Rx.T @ Ry.T @ Rz.T

    idx = np.array(list(range(0, n_centers))).reshape((args.num_figures, args.centers_fig))

    # Set up formatting for the movie files
    Writer = writers["ffmpeg"]
    writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)

    for i in range(args.num_figures):
        fig, axes = plt.subplots(3, args.centers_fig, figsize=(30, 10), subplot_kw={"projection": "3d"})
        labels_fig = list(idx[i])
        centers_fig = centers[labels_fig]

        variances = variance_per_joint(keypoints, labels=labels, filter=labels_fig)

        # TODO: error handling in case keypoints and labels do not exist
        pairs = max_distance_pairs(keypoints, labels, filter=labels_fig, viz=True)

        # Plot the centroids
        plot_poses(axes[0, 0], centers_fig[0])
        plot_poses(axes[0, 1], centers_fig[1])
        plot_poses(axes[0, 2], centers_fig[2])
        plot_poses(axes[0, 3], centers_fig[3])
        plot_poses(axes[0, 4], centers_fig[4])

        # Plot the most dissimilar poses
        plot_poses(axes[1, 0], pairs[labels_fig[0]][0], color="r")
        plot_poses(axes[1, 0], pairs[labels_fig[0]][1], color="b")
        plot_poses(axes[1, 1], pairs[labels_fig[1]][0], color="r")
        plot_poses(axes[1, 1], pairs[labels_fig[1]][1], color="b")
        plot_poses(axes[1, 2], pairs[labels_fig[2]][0], color="r")
        plot_poses(axes[1, 2], pairs[labels_fig[2]][1], color="b")
        plot_poses(axes[1, 3], pairs[labels_fig[3]][0], color="r")
        plot_poses(axes[1, 3], pairs[labels_fig[3]][1], color="b")
        plot_poses(axes[1, 4], pairs[labels_fig[4]][0], color="r")
        plot_poses(axes[1, 4], pairs[labels_fig[4]][1], color="b")

        # Plot the variance per joint
        plot_poses(axes[2, 0], centers_fig[0], color="k", c=variances[labels_fig[0]], cmap="hot")
        plot_poses(axes[2, 1], centers_fig[1], color="k", c=variances[labels_fig[1]], cmap="hot")
        plot_poses(axes[2, 2], centers_fig[2], color="k", c=variances[labels_fig[2]], cmap="hot")
        plot_poses(axes[2, 3], centers_fig[3], color="k", c=variances[labels_fig[3]], cmap="hot")
        plot_poses(axes[2, 4], centers_fig[4], color="k", c=variances[labels_fig[4]], cmap="hot")

        # Save the plot
        plt.savefig(os.path.join(args.save_dir, f"{i:02}.png"))

        # Create lines for edges in each subplot
        lines_centroids = [[ax.plot([], [], [], markersize=2)[0] for _ in range(len(SKELETON))] for ax in axes.flat[:5]]
        lines_max = [[ax.plot([], [], [], markersize=2)[0] for _ in range(2 * len(SKELETON))] for ax in axes.flat[5:10]]
        # lines_var = [[ax.plot([], [], [], markersize=2)[0] for _ in range(len(SKELETON))] for ax in axes.flat[10:]]
        lines = lines_centroids + lines_max  # + lines_var

        # Create the animation
        ani = FuncAnimation(
            fig,
            update_plot,
            frames=np.arange(0, 360, 2),
            fargs=(axes, lines, centers_fig, pairs, variances, SKELETON),
            interval=50,
        )

        # Save the animation as a video file
        ani.save(os.path.join(args.save_dir, f"{i:02}.mp4"), writer=writer)

        plt.close()
