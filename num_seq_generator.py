from typing import List, Tuple

import numpy as np
from PIL import Image


# changed function signature slightly for performance
def generate_numbers_sequence(
    digits: List,
    spacing_range: Tuple,
    image_width: int,
    data: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Generate an image that contains the sequence of given numbers, spaced
    randomly using an uniform distribution.

    Parameters
    ----------
    digits:
        A list-like containing the numerical values of the digits from which
        the sequence will be generated (for example [3, 5, 0]).
    spacing_range:
        A (minimum, maximum) pair (tuple), representing the min and max spacing
        between digits. Unit should be pixel.
    image_width:
        Specifies the width of the image in pixels.
    data:
        A dataset represented as a numpy array containing the images to use to
        generate number sequences.
    labels:
        Labels corresponding to the dataset represented as a numpy array of integers.

    Returns
    -------
    The image containing the sequence of numbers. Images should be represented
    as floating point 32bits numpy arrays with a scale ranging from 0 (black) to
    1 (white), the first dimension corresponding to the height and the second
    dimension to the width.
    """
    if len(data.shape) != 3:
        raise ValueError(
            f"Wrong number of dimensions. Expected 3 dimensions [n_samples, width, height], "
            f"found {len(data.shape)}"
        )
    if len(labels.shape) != 1:
        raise ValueError(
            f"Wrong number of dimensions. Expected 1 dimension [n_samples], found {len(labels.shape)}"
        )
    if data.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Number of samples of data and labels is different. Found {data.shape[0]} samples and "
            f"{labels.shape[0]} labels"
        )
    if not set(digits).issubset(set(labels)):
        raise ValueError("Some of the provided digits do not appear on the dataset")

    rng = np.random.default_rng()
    width = data.shape[1]
    height = data.shape[2]
    spacing = rng.integers(spacing_range[0], spacing_range[1], len(digits) - 1)
    spacing_imgs = [
        np.zeros((28, space)).astype("float32") if space != 0 else None
        for space in spacing
    ]
    img_seq = []

    for i, num in enumerate(digits):
        img = (
            rng.choice(data[labels == num], 1, replace=False)
            .reshape(width, height)
            .astype("float32")
        )
        if i == 0:
            img_seq.append(img)
        else:
            if spacing_imgs[i - 1] is not None:
                img_seq.extend([spacing_imgs[i - 1], img])
            else:
                img_seq.append(img)

    final_image = np.hstack(img_seq).astype("float32")
    im = Image.fromarray(final_image)
    im = im.resize((image_width, 28))
    img_seq = np.array(im) / 255
    img_seq = img_seq.astype("float32")
    return img_seq
