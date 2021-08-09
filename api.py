import io
from pathlib import Path
import random
from typing import List

from flask import Flask, request
from flask.helpers import send_file
from flask_cors import CORS
from PIL import Image
from waitress import serve

from get_mnist import load_mnist
from num_seq_generator import generate_numbers_sequence


app = Flask(__name__)
CORS(app)
DEFAULT_PIX_PER_NUM = 40
DEFAULT_MIN_SEQ_LEN = 1
DEFAULT_MAX_SEQ_LEN = 7
DEFAULT_MAX_SPACING = DEFAULT_PIX_PER_NUM // 8
DEFAULT_MIN_SPACING = 0
MIN_SEQ_LEN = 1
MAX_SEQ_LEN = 20
MIN_IMG_WIDTH = DEFAULT_PIX_PER_NUM
MAX_IMG_WIDTH = 1000
MIN_SPACING = 0
MAX_SPACING = 50


def input_from_query_string(args):
    if (sequence := args.get("sequence")) is not None:
        sequence = list(map(int, sequence))
    else:
        sequence = [
            random.randint(0, 9)
            for i in range(random.randint(DEFAULT_MIN_SEQ_LEN, DEFAULT_MAX_SEQ_LEN))
        ]

    if (min_spacing := args.get("min_spacing")) is not None:
        min_spacing = int(min_spacing)
    else:
        min_spacing = DEFAULT_MIN_SPACING

    if (max_spacing := args.get("max_spacing")) is not None:
        max_spacing = int(max_spacing)
    else:
        max_spacing = DEFAULT_MAX_SPACING

    if (image_width := args.get("image_width")) is not None:
        image_width = int(image_width)
    else:
        image_width = (
            len(sequence) * DEFAULT_PIX_PER_NUM
            + (len(sequence) - 1) * DEFAULT_MAX_SPACING
        )

    return sequence, min_spacing, max_spacing, image_width


def validate_generation_input(
    sequence: List[int], min_spacing: int, max_spacing: int, image_width: int
):
    if not (MIN_SEQ_LEN <= len(sequence) <= MAX_SEQ_LEN):
        raise ValueError(
            f"Wrong sequence length. Limits are [{MIN_SEQ_LEN}, {MAX_SEQ_LEN}]"
        )
    if not (MIN_SPACING <= min_spacing <= MAX_SPACING):
        raise ValueError(
            f"Wrong minimum spacing. Limits are [{MIN_SPACING}, {MAX_SPACING}]"
        )
    if not (MIN_SPACING <= max_spacing <= MAX_SPACING):
        raise ValueError(
            f"Wrong maximum spacing. Limits are [{MIN_SPACING}, {MAX_SPACING}]"
        )
    if max_spacing <= min_spacing:
        raise ValueError("Maximum spacing has to be greater than minimum space")
    if not (MIN_IMG_WIDTH <= image_width <= MAX_IMG_WIDTH):
        raise ValueError(
            f"Wrong image width. Limits are [{MIN_IMG_WIDTH}, {MAX_IMG_WIDTH}]"
        )


def serve_pil_image(pil_img: Image.Image):
    img_io = io.BytesIO()
    pil_img.save(img_io, "JPEG", quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")


@app.route("/generate")
def generate_number():
    sequence, min_spacing, max_spacing, image_width = input_from_query_string(
        request.args
    )
    validate_generation_input(sequence, min_spacing, max_spacing, image_width)

    img_seq = generate_numbers_sequence(
        sequence,
        (min_spacing, max_spacing),
        image_width,
        train_imgs,
        train_labels,
    )
    im = Image.fromarray(img_seq * 255)
    im = im.convert("L")
    return serve_pil_image(im)


if __name__ == "__main__":
    out_path = Path("data")
    if not out_path.exists():
        out_path.mkdir()
    if not out_path.is_dir():
        raise ValueError("Path to MNIST dataset is not a directory")

    train_imgs, train_labels, _, _ = load_mnist(out_path)
    serve(app, host="127.0.0.1", port=5000)
    # app.run(port=5000)
