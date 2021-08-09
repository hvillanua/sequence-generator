import io
from pathlib import Path

from flask import Flask, request
from flask.helpers import send_file
from flask_cors import CORS
from PIL import Image
from waitress import serve

from get_mnist import load_mnist
from num_seq_generator import generate_numbers_sequence


app = Flask(__name__)
CORS(app)


def validate_generation_input(min_spacing, max_spacing, image_width):
    if min_spacing < 0:
        raise ValueError("Minimum spacing can't be negative")
    if max_spacing < 0:
        raise ValueError("Maximum spacing can't be negative")
    if image_width < 1:
        raise ValueError("Image width can't be less than 1")
    if max_spacing <= min_spacing:
        raise ValueError("Maximum spacing has to be greater than minimum space")


def serve_pil_image(pil_img: Image.Image):
    img_io = io.BytesIO()
    pil_img.save(img_io, "JPEG", quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")


@app.route("/generate")
def generate_number():
    sequence = list(map(int, request.args.get("sequence")))
    min_spacing = int(request.args.get("min_spacing"))
    max_spacing = int(request.args.get("max_spacing"))
    image_width = int(request.args.get("image_width"))
    validate_generation_input(min_spacing, max_spacing, image_width)

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
