import os

import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

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

UPPER_BODY_SKELETON = [
    [0, 1],
    [0, 2],
    [0, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [4, 7],
    [7, 8],
    [8, 9],
    [4, 10],
    [10, 11],
    [11, 12],
]

angle_x = 60 * (np.pi / 180)
angle_y = 180 * (np.pi / 180)
angle_z = -30 * (np.pi / 180)

Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])


def show2Dpose(img, kps, radius=4):
    """
    Draws 2D pose keypoints on an image.

    Args:
        kps (numpy.ndarray): Array of shape (N, 2) representing the 2D keypoints.
        img (numpy.ndarray): The image on which to draw the keypoints.
        radius (int, optional): The radius of the keypoints. Defaults to 4.

    Returns:
        numpy.ndarray: The image with the keypoints drawn.

    """
    if kps.size == 0:
        return img

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j, c in enumerate(SKELETON):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=radius)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=radius)

    return img


def show3Dpose(img, kps, plot_w, plot_h, img_w):
    """
    Superimposes 3D pose keypoints on a video frame.

    Args:
        kps (numpy.ndarray): Array of shape (N, 3) representing the 3D keypoints.
        img (numpy.ndarray): The video frame on which to superimpose the keypoints.
        w (int): Width of the video frame.
        h (int): Height of the video frame.

    Returns:
        numpy.ndarray: The video frame with the 3D keypoints superimposed.

    """

    matplotlib.use("Agg")

    plt.ioff()
    fig, ax = plt.subplots(1, 1, figsize=(plot_w / 100, plot_h / 100), dpi=100, subplot_kw={"projection": "3d"})

    canvas = FigureCanvas(fig)

    kps -= kps[0]
    kps = kps @ Rx.T @ Ry.T @ Rz.T

    plot_poses(ax, kps, s=2)

    canvas.draw()

    pose_img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(plot_h, plot_w, -1)

    pose_img = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)

    # Superimpose the matplotlib figure on the video frame
    img[:plot_h, img_w - plot_w :] = pose_img

    del fig, ax, canvas

    return img


def get_upper_body(kpts):
    """
    Extracts the upper body keypoints from the full keypoints array.

    Args:
        kpts (numpy.ndarray): Array of shape (N, 17, n_dim) representing the keypoints.

    Returns:
        numpy.ndarray: The upper body keypoints array of shape (N, 8, n_dim).

    """
    N = kpts.shape[0]
    n_dim = kpts.shape[2]
    upper_body = np.zeros((N, 13, n_dim))

    for i in range(N):
        upper_body[i, 0] = kpts[i, 0]  # Hip (root joint)
        upper_body[i, 1] = kpts[i, 1]  # Right hip
        upper_body[i, 2] = kpts[i, 4]  # Left hip
        upper_body[i, 3] = kpts[i, 7]  # Spine
        upper_body[i, 4] = kpts[i, 8]  # Neck
        upper_body[i, 5] = kpts[i, 9]  # Nose
        upper_body[i, 6] = kpts[i, 10]  # Head
        upper_body[i, 7] = kpts[i, 11]  # Left shoulder
        upper_body[i, 8] = kpts[i, 12]  # Left elbow
        upper_body[i, 9] = kpts[i, 13]  # Left wrist
        upper_body[i, 10] = kpts[i, 14]  # Right shoulder
        upper_body[i, 11] = kpts[i, 15]  # Right elbow
        upper_body[i, 12] = kpts[i, 16]  # Right wrist

    return upper_body


def get_video_metadata(video):
    """
    Retrieves metadata information from a video file.

    Parameters:
    - video (str): The path to the video file.

    Returns:
    - dict: A dictionary containing the following metadata information:
        - 'fps': Frames per second of the video.
        - 'video_length': Total number of frames in the video.
        - 'width': Width of the video frames.
        - 'height': Height of the video frames.
    """
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    cv2.destroyAllWindows()

    return {"fps": fps, "video_length": video_length, "width": width, "height": height}


def video_to_frames(video, percentage=1.0, random=True):
    """
    Extracts frames from a video file and returns a list of frames.

    Parameters:
    - video (str): The path to the video file.
    - percentage (float): The percentage of frames to extract from the video. Default is 1.0 (all frames).
    - random (bool): Whether to extract frames randomly or sequentially. Default is True.

    Returns:
    - frames (list): A list of frames extracted from the video.

    If the `percentage` parameter is less than 1.0, a subset of frames will be extracted based on the percentage.
    If `random` is True, the frames will be randomly selected. Otherwise, the frames will be selected sequentially.
    The function uses OpenCV to read the video file and extract frames.

    Note: Make sure to have OpenCV installed in your environment before using this function.
    """

    cap = cv2.VideoCapture(video)
    frames = []

    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    if percentage < 1.0:
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = int(video_length * percentage)
        frame_idxs = np.random.randint(0, video_length - 1, (n,)) if random else np.arange(0, n)
        frames = frames[frame_idxs]

        return frames, frame_idxs
    return frames


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


def plot_poses(ax, pose, color=None, c=None, cmap=None, title=None, ticks=False, s=7):
    """
    Plots 3D poses on a given axis.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis on which to plot the poses.
    - pose (numpy.ndarray): The 3D pose data to be plotted.
    - color (str): The color of the pose points and lines. Default is None.
    - c (array-like): The color values for the pose points. Default is None.
    - cmap (str or matplotlib.colors.Colormap): The colormap for the pose points. Default is None.
    - title (str): The title of the plot. Default is None.
    - ticks (bool): Whether to display ticks on the plot. Default is True.
    - s (float): The size of the pose points. Default is 5.

    Returns:
    None
    """

    for joint in SKELETON:
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=c, cmap=cmap, s=s)
        ax.plot(
            [pose[joint[0], 0], pose[joint[1], 0]],
            [pose[joint[0], 1], pose[joint[1], 1]],
            zs=[pose[joint[0], 2], pose[joint[1], 2]],
            color=color,
        )
        ax.view_init(azim=80)
        ax.set_title(title)

        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])


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
        if samples:  # If there are samples for the class
            k0, k1 = get_extreme_samples(samples, mode="distant", viz=True)
            pairs[k] = (k0, k1)
        else:
            pairs[k] = (np.zeros((17, 3)), np.zeros((17, 3)))

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
        if kpts:  # If there are samples for the class
            covs = []
            for i in range(17):
                joint = np.array([kpt.reshape(17, 3)[i] for kpt in kpts])

                # Compute determinant of covariance matrix as a measure of general variance
                cov_det = np.linalg.det(np.cov(joint, rowvar=False))
                covs.append(cov_det)

            covariances_per_class[k] = covs
        else:
            covariances_per_class[k] = [0] * 17

    return covariances_per_class
