import numpy as np
from PIL import Image


# changed function signature slightly for performance
def generate_numbers_sequence(digits, spacing_range, image_width, data, labels):
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
        raise ValueError(f"Wrong number of dimensions. Expected 3 dimensions [n_samples, width, height], "
                         f"found {len(data.shape)}")
    if len(labels.shape) != 1:
        raise ValueError(f"Wrong number of dimensions. Expected 1 dimension [n_samples], found {len(labels.shape)}")
    if data.shape[0] != labels.shape[0]:
        raise ValueError(f"Number of samples of data and labels is different. Found {data.shape[0]} samples and "
                         f"{labels.shape[0]} labels")

    rng = np.random.default_rng()
    width = data.shape[1]
    height = data.shape[2]
    spacing = rng.integers(spacing_range[0], spacing_range[1], len(digits) - 1)
    spacing_imgs = [np.zeros((28, space)) if space != 0 else None for space in spacing]

    for i, num in enumerate(digits):
        img = rng.choice(data[labels == num], 1, replace=False).reshape(width, height).astype("float32")
        img = img / 255
        # print(img.shape, img.min(), img.max(), img.dtype)
        if i == 0:
            img_seq = img
        else:
            if spacing_imgs[i-1] is not None:
                img_seq = np.hstack((img_seq, spacing_imgs[i-1], img)).astype("float32")
            else:
                img_seq = np.hstack((img_seq, img)).astype("float32")
        # print(img_seq.shape, img_seq.min(), img_seq.max(), img_seq.dtype)

    print(img_seq.shape)
    # TODO: resize image
    # img_seq.resize((28, image_width))
    print(img_seq.shape)
    return img_seq
