""" Perform clusterin on the 3D poses using classical algorithms """

import os
import glob
import argparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints", type=str, required=True, help="file path to keypoinys npz")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert os.path.isfile(args.keypoints), f"{args.keypoints} does not exist"

    x = np.load(args.keypoints)["keypoints"].reshape((-1, 17 * 3))
    x = x[1000:-1000, :]  # cropping initial and final parts of the video

    # candidate values for our number of cluster
    parameters = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]
    # instantiating ParameterGrid, pass number of clusters as input
    parameter_grid = ParameterGrid({"n_clusters": parameters})
    best_score = -1
    kmeans_model = KMeans()  # instantiating KMeans model
    silhouette_scores = []
    # evaluation based on silhouette_score
    for p in parameter_grid:
        kmeans_model.set_params(**p)  # set current hyper parameter
        kmeans_model.fit(x)  # fit model on wine dataset, this will find clusters based on parameter p
        ss = metrics.silhouette_score(x, kmeans_model.labels_)  # calculate silhouette_score
        silhouette_scores += [ss]  # store all the scores
        print("Parameter:", p, "Score", ss)
        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p
    # plotting silhouette score
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align="center", color="#722f59", width=0.5)
    plt.xticks(range(len(silhouette_scores)), list(parameters))
    plt.title("Silhouette Score", fontweight="bold")
    plt.xlabel("Number of Clusters")
    plt.show()
