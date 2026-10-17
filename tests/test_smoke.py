"""Smoke tests.

Most of these need neither the dataset nor a trained model, so they run in
seconds and catch the failure modes that actually bite in this project:

  - the model silently breaking at a non-256px input size
  - remedies.json drifting out of sync with the dataset's class names
  - inference preprocessing drifting away from training preprocessing
  - the web app 500-ing instead of explaining that no model exists yet

Run:  python -m pytest tests/ -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.data import build_transforms, discover_class_names
from src.model import ResNet6, ResNet9, build_model, count_parameters
from src.remedies import (coverage, display_name, is_healthy, load_remedies,
                          prettify)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@pytest.mark.parametrize("img_size", [64, 128, 256])
def test_resnet9_accepts_every_supported_size(img_size):
    """The fixed-MaxPool(4) variant crashes at 64px; ours must not."""
    model = ResNet9(num_classes=38)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, img_size, img_size))
    assert out.shape == (2, 38)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("img_size", [64, 128, 256])
def test_resnet6_accepts_every_supported_size(img_size):
    model = ResNet6(num_classes=38)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, img_size, img_size))
    assert out.shape == (2, 38)


def test_resnet9_is_deeper_than_resnet6():
    assert count_parameters(ResNet9()) > count_parameters(ResNet6())


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model("resnet42")


def test_model_honours_num_classes():
    model = build_model("resnet9", num_classes=7)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape[1] == 7


def test_single_sample_batch_does_not_break_batchnorm_in_training():
    """At 64px the deepest feature map is 1x1, and BatchNorm raises on a
    batch of 1 in train mode. build_loaders must never hand the loop such a
    batch - see the drop_last logic in src/data.py."""
    from src.data import should_drop_last
    assert should_drop_last(n=70295, batch_size=32) is False   # remainder 7
    assert should_drop_last(n=65, batch_size=32) is True       # remainder 1
    assert should_drop_last(n=1, batch_size=32) is False       # degenerate


def test_logits_are_not_probabilities():
    """Softmax must be applied at inference, not inside the model, because
    cross_entropy expects raw logits - applying it twice quietly hurts."""
    model = ResNet9(num_classes=38)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 64, 64))
    sums = out.sum(dim=1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
def test_validation_transform_is_deterministic():
    """Inference must give the same tensor every time for the same image."""
    from PIL import Image
    img = Image.new("RGB", (300, 220), (120, 180, 90))
    tf = build_transforms(64, train=False)
    assert torch.equal(tf(img), tf(img))


def test_transform_output_shape_and_range():
    from PIL import Image
    img = Image.new("RGB", (300, 220), (120, 180, 90))
    for size in (64, 128, 256):
        t = build_transforms(size, train=False)(img)
        assert t.shape == (3, size, size)
        # The paper normalises to [0, 1] via ToTensor only - no mean/std.
        assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


def test_training_transform_actually_augments():
    from PIL import Image
    img = Image.new("RGB", (300, 220), (120, 180, 90))
    # Put some structure in, or random crops of a flat colour look identical.
    for x in range(0, 300, 20):
        for y in range(0, 220, 20):
            img.putpixel((x, y), (255, 0, 0))
    tf = build_transforms(64, train=True, augment=True)
    assert not torch.equal(tf(img), tf(img))


def test_augmentation_can_be_disabled():
    from PIL import Image
    img = Image.new("RGB", (120, 120), (10, 200, 10))
    tf = build_transforms(64, train=True, augment=False)
    assert torch.equal(tf(img), tf(img))


# --------------------------------------------------------------------------
# remedies
# --------------------------------------------------------------------------
def test_remedies_cover_all_38_classes():
    remedies = load_remedies()
    assert len(remedies) == config.NUM_CLASSES


def test_every_remedy_has_the_required_fields():
    required = {"crop", "disease", "healthy", "pathogen", "symptoms",
                "organic", "chemical", "prevention"}
    for name, rec in load_remedies().items():
        assert set(rec) == required, "%s has fields %s" % (name, sorted(rec))
        for field in ("crop", "disease", "symptoms", "organic", "chemical",
                      "prevention"):
            assert isinstance(rec[field], str) and rec[field].strip(), \
                "%s.%s is empty" % (name, field)


def test_remedy_keys_match_the_dataset_exactly():
    """The dataset's folder names include a misspelling (Haunglongbing),
    embedded spaces and a trailing underscore. A mismatch here would surface as
    a wrong remedy at inference, so it is asserted rather than trusted."""
    if not config.TRAIN_DIR.is_dir():
        pytest.skip("dataset not extracted")
    missing, extra = coverage(discover_class_names())
    assert not missing, "classes with no remedy: %s" % missing
    assert not extra, "remedies for non-existent classes: %s" % extra


def test_healthy_classes_are_flagged_consistently():
    for name, rec in load_remedies().items():
        assert rec["healthy"] == name.lower().endswith("healthy"), name
        if rec["healthy"]:
            assert rec["pathogen"] is None, name


def test_diseased_classes_name_a_pathogen():
    for name, rec in load_remedies().items():
        if not rec["healthy"]:
            assert rec["pathogen"], "%s has no pathogen" % name


def test_there_are_26_diseases_as_the_paper_states():
    diseased = [n for n, r in load_remedies().items() if not r["healthy"]]
    assert len(diseased) == 26


def test_display_names_are_human_readable():
    assert display_name("Tomato___Early_blight") == "Tomato - Early Blight"
    assert "Huanglongbing" in display_name("Orange___Haunglongbing_(Citrus_greening)")
    assert is_healthy("Tomato___healthy")
    assert not is_healthy("Tomato___Early_blight")


def test_unknown_class_does_not_raise():
    from src.remedies import lookup
    rec = lookup("Banana___Some_New_Disease")
    assert rec["known"] is False
    assert rec["disease"]


def test_prettify_handles_dataset_quirks():
    assert prettify("Corn_(maize)___Common_rust_") == "Corn (maize) - Common rust"
    assert prettify("Pepper,_bell___healthy").startswith("Pepper, bell")


# --------------------------------------------------------------------------
# dataset (skipped when not extracted)
# --------------------------------------------------------------------------
def test_dataset_has_the_expected_shape():
    if not config.TRAIN_DIR.is_dir():
        pytest.skip("dataset not extracted")
    assert len(discover_class_names()) == config.NUM_CLASSES
    n_train = sum(1 for _ in config.TRAIN_DIR.rglob("*") if _.is_file())
    assert n_train == config.EXPECTED_TRAIN


def test_no_leftover_part_files():
    """The extractor writes .part then renames, so none should survive."""
    if not config.RAW_DIR.is_dir():
        pytest.skip("dataset not extracted")
    leftovers = list(config.RAW_DIR.rglob("*.part"))
    assert not leftovers, "interrupted extraction left %d .part files" % len(leftovers)


# --------------------------------------------------------------------------
# web app
# --------------------------------------------------------------------------
@pytest.fixture
def client():
    from app.app import app as flask_app
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as c:
        yield c


def test_index_responds(client):
    resp = client.get("/")
    # 200 with a model, 503 with setup instructions without one - never a 500.
    assert resp.status_code in (200, 503)
    assert b"Plant Disease Detection" in resp.data


def test_health_endpoint_reports_model_state(client):
    resp = client.get("/health")
    assert resp.status_code in (200, 503)
    body = json.loads(resp.data)
    assert "model_loaded" in body
    assert body["status"] in ("ok", "model_not_trained")


def test_predict_without_a_file_is_a_clean_error(client):
    resp = client.post("/predict", data={})
    assert resp.status_code in (400, 503)
    assert b"Internal Server Error" not in resp.data


def test_api_rejects_a_non_image(client):
    import io as _io
    resp = client.post("/api/predict", data={
        "image": (_io.BytesIO(b"this is not an image"), "notes.txt")})
    assert resp.status_code in (400, 503)
    assert "error" in json.loads(resp.data)


# --------------------------------------------------------------------------
# end-to-end (only when a model exists)
# --------------------------------------------------------------------------
def test_trained_model_predicts_a_test_image():
    if not config.MODEL_FILE.is_file():
        pytest.skip("no trained model yet")
    if not config.TEST_DIR.is_dir():
        pytest.skip("dataset not extracted")
    from src.inference import load_image, load_model, predict_image

    model, class_names, img_size = load_model()
    images = sorted(p for p in config.TEST_DIR.iterdir() if p.is_file())
    if not images:
        pytest.skip("no test images")

    res = predict_image(model, load_image(images[0]), class_names, img_size)
    assert res["class_name"] in class_names
    assert 0.0 <= res["confidence"] <= 1.0
    # Probabilities over the shortlist must be ordered and normalised.
    confs = [p["confidence"] for p in res["predictions"]]
    assert confs == sorted(confs, reverse=True)
    assert res["remedy"]["display"]


def test_checkpoint_records_its_image_size():
    """Inference reads img_size from the checkpoint; without it the app would
    silently preprocess at the wrong resolution."""
    if not config.MODEL_FILE.is_file():
        pytest.skip("no trained model yet")
    ckpt = torch.load(config.MODEL_FILE, map_location="cpu", weights_only=False)
    assert ckpt["img_size"] in (64, 128, 256)
    assert len(ckpt["class_names"]) == config.NUM_CLASSES
    assert ckpt["model"] in ("resnet9", "resnet6")
