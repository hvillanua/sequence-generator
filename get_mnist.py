from pathlib import Path
from urllib import request
import gzip
import pickle

import numpy as np


# Although I made some changes to conform to PEP8 and add some more flexibility
# attribution to the original code: https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py
_mnist_files = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


def _download_mnist(data_path):
    # TODO: add function explanation
    base_url = "http://yann.lecun.com/exdb/mnist/"
    downloaded = False
    for name in _mnist_files:
        file_path = data_path / name[1]
        if not file_path.exists():
            print(f"Downloading, {name[1]}. This process may take a few minutes...")
            request.urlretrieve(base_url + name[1], data_path / name[1])
            downloaded = True
    if downloaded:
        print("Download complete.")


def _save_mnist(data_path):
    # TODO: add function explanation
    mnist = {}
    for name in _mnist_files[:2]:
        with gzip.open(data_path / name[1], "rb") as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    for name in _mnist_files[-2:]:
        with gzip.open(data_path / name[1], "rb") as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    with open(data_path / "mnist.pkl", "wb") as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def load_mnist(data_path):
    # TODO: add function explanation
    if not data_path.is_dir():
        raise ValueError("Data path is not a directory")

    mnist_pkl = data_path / "mnist.pkl"
    if not mnist_pkl.exists():
        _download_mnist(data_path)
        _save_mnist(data_path)
    with open(mnist_pkl, "rb") as f:
        mnist = pickle.load(f)

    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
