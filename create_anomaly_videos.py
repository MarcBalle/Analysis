import os
import argparse
import pickle
import glob
import warnings

import cv2
import numpy as np
from tqdm import tqdm

from utils.utils import show2Dpose, show3Dpose, get_upper_body


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints_dir", type=str, required=True, help="directory containing the data")
    parser.add_argument("--video_dir", type=str, required=True, help="directory containing the videos")
    parser.add_argument("--model", type=str, required=True, help="file path to the trained model")
    parser.add_argument("--output", type=str, default=".", help="output directory path")
    parser.add_argument("--three_dimensional", action="store_true", help="visualize 3D poses")
    parser.add_argument("--only_upper_body", action="store_true", help="use only upper body keypoints")

    return parser.parse_args()


if __name__ == "__main__":
    np.random.seed(42)

    args = argument_parser()

    if args.only_upper_body:
        if not args.model.__contains__("upper"):
            warnings.warn("The model might has not been trained with only upper body keypoints.")

    driving = glob.glob(os.path.join(args.keypoints_dir, "*driving*"))
    anomaly = glob.glob(os.path.join(args.keypoints_dir, "*anomaly*"))

    model = pickle.load(open(args.model, "rb"))

    # Randomly select 5 driving and 5 anomaly videos
    driving = np.random.choice(driving, 5)
    anomaly = np.random.choice(anomaly, 5)

    video_files = {"driving": driving, "anomaly": anomaly}

    threshold = model.threshold_

    for key in video_files.keys():
        filepaths = video_files[key]
        os.makedirs(os.path.join(args.output, key), exist_ok=True)

        print(f"Creating {key} videos...")
        for i, f in tqdm(enumerate(filepaths)):
            video_name = os.path.basename(f).replace("npz", "mp4")

            kpts = np.load(f)["keypoints"]

            if args.only_upper_body:
                kpts_whole_boddy = kpts.copy()  # Save whole body keypoints for visualization
                kpts = get_upper_body(kpts)

            cap = cv2.VideoCapture(os.path.join(args.video_dir, video_name))
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            scores = model.decision_function(kpts.reshape((kpts.shape[0], -1)))

            assert len(scores) == video_length

            out = cv2.VideoWriter(
                filename=os.path.join(args.output, key, video_name),
                fourcc=cv2.VideoWriter_fourcc(*"DIVX"),
                fps=fps,
                frameSize=(int(width), int(height)),
            )

            for i in range(video_length):
                score = scores[i]
                ret, img = cap.read()
                if not ret:
                    break

                if score > threshold:
                    color = (0, 0, 255)  # Red color in BGR
                else:
                    color = (0, 255, 0)  # Green color in BGR

                # Draw a rectangle around the whole frame
                start_point = (0, 0)
                end_point = (width - 1, height - 1)
                thickness = 2  # Thickness of the rectangle border in pixels
                img = cv2.rectangle(img, start_point, end_point, color, thickness)

                # Show pose on the frame
                if args.three_dimensional:
                    img = show3Dpose(
                        img,
                        kpts_whole_boddy[i] if args.only_upper_body else kpts[i],
                        plot_w=width // 4,
                        plot_h=height // 2,
                        img_w=width,
                    )
                else:
                    img = show2Dpose(
                        img,
                        kpts_whole_boddy[i] if args.only_upper_body else kpts[i],
                    )

                out.write(img)

            cap.release()
            out.release()
            cv2.destroyAllWindows()
