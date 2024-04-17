""" Cropping a video and its corresponding keypoints array """

import os
import argparse

import cv2
import numpy as np
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="path to video file")
    parser.add_argument("--keypoints", type=str, required=True, help="keypoints npz file path")
    parser.add_argument(
        "--start_time", type=int, required=True, help="starting point in the video where to crop (given in seconds)"
    )
    parser.add_argument(
        "--end_time", type=int, required=True, help="ending point in the video where to crop (given in seconds)"
    )
    parser.add_argument("--out_name", type=str, required=True, help="save name for the cropped video and keypoints")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    start_frame = int(args.start_time * fps)
    end_frame = int(args.end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    out = cv2.VideoWriter(
        filename=os.path.join(".", "output", args.out_name + ".mp4"),
        fourcc=cv2.VideoWriter_fourcc(*"DIVX"),
        fps=fps,
        frameSize=(int(width), int(height)),
    )

    for i in tqdm(range(end_frame - start_frame)):
        _, img = cap.read()
        out.write(img)

    keypoints = np.load(args.keypoints)["keypoints"]
    keypoints = keypoints[start_frame:end_frame]

    np.savez(os.path.join(".", "output", args.out_name), keypoints=keypoints)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
