import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints", type=str, required=True, help="file path to keypoinys npz")
    parser.add_argument("--output", type=str, required=False, help="output directory")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    assert os.path.isfile(args.keypoints), f"{args.keypoints} does not exist"

    keypoints = np.load(args.keypoints)["arr_0"]  # TODO: reshape to flat array if needed
    N = keypoints.shape[0]

    perplexities = np.random.randint(5, N - int(N / 6), (10,))

    colors = np.arange(keypoints.shape[0])

    for perpelexity in perplexities:
        tsne = TSNE(n_components=2, perplexity=perpelexity)
        keypoints_embedded = tsne.fit_transform(keypoints)

        plt.scatter(keypoints_embedded[:, 0], keypoints_embedded[:, 1], c=colors, cmap="viridis")
        # Add a label to each point
        for i in range(keypoints_embedded.shape[0]):
            plt.text(keypoints_embedded[i, 0], keypoints_embedded[i, 1], f"{i}")

        plt.title(f"TSNE embedding with perplexity={perpelexity}")
        plt.savefig(os.path.join(args.output, f"tsne_{perpelexity}.png"))
        plt.close()

        np.savez(os.path.join(args.output, f"tsne_{perpelexity}"), keypoints_embedded)
