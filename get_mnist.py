from pathlib import Path
from urllib import request
import gzip
import pickle

import numpy as np


# Although I made some changes to conform to PEP8 and add some more flexibility
# attribution to the original code: https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py
_filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


def _download_mnist(data_path):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in _filename:
        print(f"Downloading, {name[1]}. This process may take a few minutes...")
        request.urlretrieve(base_url + name[1], data_path / name[1])
    print("Download complete.")


def _save_mnist(data_path):
    mnist = {}
    for name in _filename[:2]:
        with gzip.open(name[1], "rb") as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)

    for name in _filename[-2:]:
        with gzip.open(name[1], "rb") as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    with open(data_path / "mnist.pkl", "wb") as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def load_mnist(data_path=Path("data")):
    if not data_path.exists():
        data_path.mkdir()
    if not data_path.is_dir():
        raise ValueError("Data path is not a directory")

    mnist_file = data_path / "mnist.pkl"
    if not mnist_file.exists():
        print("No existing mnist file found, download will start now.")
        _download_mnist(data_path)
        _save_mnist(data_path)
    with open(mnist_file, "rb") as f:
        print("Existing mnist file found, proceeding to load.")
        mnist = pickle.load(f)

    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
