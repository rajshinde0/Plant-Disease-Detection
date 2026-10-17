"""Flask web app for plant disease detection.

Runs entirely offline: the model is a local .pth file and remedies come from
data/remedies.json, so the app works with no internet connection at all. That
is the paper's stated requirement - farmers in remote areas need a remedy
exactly where connectivity is worst.

Preprocessing is imported from src.data via src.inference rather than being
redefined here, so inference can never drift away from how the model was
trained.

Run:
    python app/app.py
    -> http://127.0.0.1:5000
"""
from __future__ import annotations

import io
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.inference import (ModelNotTrained, checkpoint_info, load_model,
                           predict_image)
from src.remedies import load_meta

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES

# Loaded lazily and cached, so startup does not fail when no model exists yet.
_STATE: dict = {}


def get_model():
    """Return (model, class_names, img_size), loading once per process."""
    if "model" not in _STATE:
        model, class_names, img_size = load_model()
        _STATE["model"] = model
        _STATE["class_names"] = class_names
        _STATE["img_size"] = img_size
        _STATE["info"] = checkpoint_info()
    return _STATE["model"], _STATE["class_names"], _STATE["img_size"]


def model_info() -> dict:
    return _STATE.get("info", {})


def allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in config.ALLOWED_UPLOAD_EXTENSIONS


@app.route("/")
def index():
    try:
        get_model()
    except ModelNotTrained as exc:
        return render_template("no_model.html", message=str(exc)), 503
    return render_template("index.html", info=model_info(),
                           meta=load_meta(), samples=sample_images())


def sample_images(limit: int = 6):
    """A few bundled test images, so the app is demoable without a file."""
    if not config.TEST_DIR.is_dir():
        return []
    files = sorted(p.name for p in config.TEST_DIR.iterdir() if p.is_file())
    return files[:limit]


def _render_error(message: str, code: int = 400):
    return render_template("index.html", error=message, info=model_info(),
                           meta=load_meta(), samples=sample_images()), code


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model, class_names, img_size = get_model()
    except ModelNotTrained as exc:
        return render_template("no_model.html", message=str(exc)), 503

    # Either an uploaded file, or one of the bundled sample images.
    sample = request.form.get("sample")
    if sample:
        safe = secure_filename(sample)
        path = config.TEST_DIR / safe
        if not path.is_file():
            return _render_error("Sample image not found: %s" % safe, 404)
        try:
            image = Image.open(path).convert("RGB")
        except (OSError, UnidentifiedImageError):
            return _render_error("Could not read that sample image.", 400)
        stored_name = safe
        saved = config.UPLOAD_DIR / safe
        if not saved.exists():
            config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            image.save(saved)
    else:
        if "image" not in request.files:
            return _render_error("No file was submitted.")
        upload = request.files["image"]
        if not upload.filename:
            return _render_error("Please choose an image first.")
        if not allowed(upload.filename):
            return _render_error(
                "Unsupported file type. Use JPG, PNG, BMP or WEBP.")
        raw = upload.read()
        if not raw:
            return _render_error("That file is empty.")
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except (OSError, UnidentifiedImageError):
            return _render_error(
                "That file is not a readable image. Try a different photo.")

        stored_name = "%s%s" % (uuid.uuid4().hex[:12],
                                Path(secure_filename(upload.filename)).suffix.lower())
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        image.save(config.UPLOAD_DIR / stored_name)

    t0 = time.time()
    result = predict_image(model, image, class_names, img_size, topk=3)
    elapsed_ms = (time.time() - t0) * 1000

    return render_template("result.html", result=result, image_name=stored_name,
                           elapsed_ms=elapsed_ms, info=model_info(),
                           meta=load_meta(),
                           threshold=int(config.LOW_CONFIDENCE_THRESHOLD * 100))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint. Accepts multipart form-data with an `image` field."""
    try:
        model, class_names, img_size = get_model()
    except ModelNotTrained as exc:
        return jsonify({"error": "model_not_trained", "detail": str(exc)}), 503

    if "image" not in request.files:
        return jsonify({"error": "no_file",
                        "detail": "POST multipart form-data with an 'image' field"}), 400
    upload = request.files["image"]
    if not upload.filename or not allowed(upload.filename):
        return jsonify({"error": "unsupported_type",
                        "detail": "use jpg, png, bmp or webp"}), 400
    try:
        image = Image.open(io.BytesIO(upload.read())).convert("RGB")
    except (OSError, UnidentifiedImageError):
        return jsonify({"error": "unreadable_image"}), 400

    t0 = time.time()
    result = predict_image(model, image, class_names, img_size,
                           topk=int(request.args.get("topk", 3)))
    result["inference_ms"] = round((time.time() - t0) * 1000, 2)
    return jsonify(result)


@app.route("/health")
def health():
    try:
        get_model()
        ready = True
        detail = None
    except ModelNotTrained as exc:
        ready = False
        detail = str(exc)
    return jsonify({
        "status": "ok" if ready else "model_not_trained",
        "model_loaded": ready,
        "detail": detail,
        "info": model_info(),
    }), (200 if ready else 503)


@app.errorhandler(413)
def too_large(_exc):
    mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
    return _render_error("That image is larger than %d MB. "
                        "Please upload a smaller photo." % mb, 413)


@app.route("/favicon.ico")
def favicon():
    # No icon shipped; 204 keeps the browser from logging a 404 on every page.
    return "", 204


if __name__ == "__main__":
    print("Plant Disease Detection - starting Flask app")
    try:
        _m, names, size = get_model()
        info = model_info()
        print("  model    : %s" % config.MODEL_FILE.name)
        print("  classes  : %d" % len(names))
        print("  img size : %d" % size)
        if info.get("val_acc") is not None:
            print("  val acc  : %.2f%%" % (info["val_acc"] * 100))
    except ModelNotTrained:
        print("  WARNING: no trained model found yet.")
        print("  The app will start but show setup instructions until you run:")
        print("    python -m src.train --img-size 64 --subset 0.3 --epochs 4")
    print()
    print("  open http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
