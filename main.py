import argparse

import numpy as np

from get_mnist import load_mnist


def generate_numbers_sequence(digits, spacing_range, image_width):
    """
    Generate an image that contains the sequence of given numbers, spaced
    randomly using an uniform distribution.

    Parameters
    ----------
    digits:
    A list-like containing the numerical values of the digits from which
        the sequence will be generated (for example [3, 5, 0]).
    spacing_range:
    a (minimum, maximum) pair (tuple), representing the min and max spacing
        between digits. Unit should be pixel.
    image_width:
        specifies the width of the image in pixels.

    Returns
    -------
    The image containing the sequence of numbers. Images should be represented
    as floating point 32bits numpy arrays with a scale ranging from 0 (black) to
    1 (white), the first dimension corresponding to the height and the second
    dimension to the width.
    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", help="Number to be generated", type=int)
    parser.add_argument("min_spacing", help="Minimum pixel spacing between consecutive digits", type=int)
    parser.add_argument("max_spacing", help="Maximum pixel spacing between consecutive digits", type=int)
    parser.add_argument("image_width", help="Pixel width of the generated image", type=int)
    args = parser.parse_args()

    if args.min_spacing < 0:
        raise ValueError("Minimum spacing can't be negative")
    if args.max_spacing < 0:
        raise ValueError("Minimum spacing can't be negative")
    if args.image_width < 1:
        raise ValueError("Image width can't be less than 1")

    generate_numbers_sequence(args.sequence, (args.min_spacing, args.max_spacing), args.image_width)
    print(load_mnist())
