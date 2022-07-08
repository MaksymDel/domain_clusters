import argparse
import os
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.cluster import KMeans, MiniBatchKMeans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedded-chunk-paths",
        type=str, nargs="+",
        help="Paths to files containing chunks of the dataset with sentence representations",
    )
    parser.add_argument(
        "--out-file-model", type=str, help="Path to file to save kmeans model"
    )
    parser.add_argument(
        "--out-file-labels",
        type=str,
        help="Path to file to save cluster labels for the whole dataset",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=8, help="Number of clusters of k-means"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="For how many epochs to run the k-means algorithm?",
    )
    parser.add_argument(
        "--verbose", type=int, default=0, help="Verbosity"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state"
    )
    args = parser.parse_args()


    model = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        compute_labels=True,
        verbose=args.verbose,
        random_state=args.random_state,
    )

    # fit
    for i_epoch in range(args.num_epochs):
        print(f"{i_epoch=}")
        for subdataset_path in args.embedded_chunk_paths:
            features = np.load(subdataset_path)["arr_0"]
            model.partial_fit(features)
            del features
            print(f"{model.inertia_=}")
    Path(os.path.dirname(args.out_file_model)).mkdir(parents=True, exist_ok=True)
    dump(model, args.out_file_model)
    print(f"Saved model to {args.out_file_model}")

    # label
    all_labels = []
    for subdataset_path in args.embedded_chunk_paths:
        features = np.load(subdataset_path)["arr_0"]
        all_labels.extend(model.predict(features))
        del features
        
    Path(os.path.dirname(args.out_file_labels)).mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out_file_labels, np.asarray(all_labels), fmt="%i")
    print(f"Saved labels to {args.out_file_labels}")
