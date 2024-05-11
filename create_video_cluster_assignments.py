""" 
This script creates a video where each in each frame 
the centroid of the cluster assigned to the particular frame is plotted. 

Besides, a T-SNE plot is created to visualize the clusters in 2D
"""

import argparse
import os
import pickle

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

from utils.utils import video_to_frames, get_video_metadata, plot_poses, Rx, Ry, Rz

matplotlib.use("Agg")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="file path to video")
    parser.add_argument("--centroids", type=str, required=True, help="file path to npz containing centers information")
    parser.add_argument("--keypoints", type=str, required=True, help="file path to npz containing keypoints")
    parser.add_argument("--labels", type=str, required=True, help="file path to npz containing labels information")
    parser.add_argument(
        "--keypoints_to_video",
        type=str,
        required=True,
        help="file path to pickle containing keypoints to video mapping",
    )
    parser.add_argument("--tsne", type=str, required=True, help="file path to tsne npz")
    parser.add_argument("--output", type=str, required=False, help="output directory")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    assert os.path.isfile(args.video), f"{args.video} does not exist"
    assert os.path.isfile(args.centroids), f"{args.centroids} does not exist"
    assert os.path.isfile(args.keypoints), f"{args.keypoints} does not exist"
    assert os.path.isfile(args.tsne), f"{args.tsne} does not exist"
    assert os.path.isfile(args.labels), f"{args.labels} does not exist"
    assert os.path.isfile(args.keypoints_to_video), f"{args.keypoints_to_video} does not exist"

    # Load the data
    # frames, frame_idxs = video_to_frames(args.video, percentage=0.5, random=False)
    centroids = np.load(args.centroids)["arr_0"].reshape((-1, 17, 3))
    keypoints = np.load(args.keypoints)["arr_0"].reshape((-1, 17, 3))
    labels = np.load(args.labels)["arr_0"]
    tsne = np.load(args.tsne)["arr_0"]
    keypoints_to_video = pickle.load(open(args.keypoints_to_video, "rb"))

    # Get the labels for the video frames
    video_name = os.path.basename(args.video)
    video_labels = labels[keypoints_to_video[video_name]]
    video_kpts = keypoints[keypoints_to_video[video_name]]

    # Create a video with the centroids
    metadata = get_video_metadata(args.video)
    fps, video_length, width, height = metadata["fps"], metadata["video_length"], metadata["width"], metadata["height"]
    out = cv2.VideoWriter(
        filename=os.path.join(args.output, "tsne_" + video_name),
        fourcc=cv2.VideoWriter_fourcc(*"DIVX"),
        fps=fps,
        frameSize=(width, height),
    )

    plot_width = width // 4
    plot_height = height // 2

    cap = cv2.VideoCapture(args.video)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if i % 1000 == 0:
            print(f"Frame {i}/{video_length}")

        if ret:
            label = video_labels[i]
            centroid = centroids[label]
            kpt = video_kpts[i]

            plt.ioff()
            fig0, ax0 = plt.subplots(
                1, 1, figsize=(plot_width / 100, plot_height / 100), dpi=100, subplot_kw={"projection": "3d"}
            )
            fig1, ax1 = plt.subplots(1, 1, figsize=(plot_width / 100, plot_height / 100), dpi=100)

            canvas0 = FigureCanvas(fig0)
            canvas1 = FigureCanvas(fig1)

            centroid -= centroid[0]
            centroid = centroid @ Rx.T @ Ry.T @ Rz.T

            kpt -= kpt[0]
            kpt = kpt @ Rx.T @ Ry.T @ Rz.T

            plot_poses(ax0, centroid, s=2)
            plot_poses(ax0, kpt, s=2, color="magenta")

            for j, emb in enumerate(tsne):
                ax1.scatter(emb[0], emb[1], color="r" if j == label else "b", s=30 if j == label else 10)
                ax1.set_xticks([])
                ax1.set_yticks([])

            canvas0.draw()
            canvas1.draw()

            plot_img0 = np.frombuffer(canvas0.tostring_rgb(), dtype="uint8").reshape(plot_height, plot_width, -1)
            plot_img1 = np.frombuffer(canvas1.tostring_rgb(), dtype="uint8").reshape(plot_height, plot_width, -1)

            plot_img0 = cv2.cvtColor(plot_img0, cv2.COLOR_RGB2BGR)
            plot_img1 = cv2.cvtColor(plot_img1, cv2.COLOR_RGB2BGR)

            # Superimpose the matplotlib figure on the video frame
            frame[:plot_height, width - plot_width :] = plot_img0
            frame[height - plot_height :, width - plot_width :] = plot_img1

            out.write(frame)

            del fig0, fig1, ax0, ax1, canvas0, canvas1

            i += 1
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
