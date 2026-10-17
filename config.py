"""Central configuration for the Plant Disease Detection project.

Every path is derived from this file's location, so the project folder can be
copied to another machine (e.g. the CUDA laptop) without edits.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- data ---------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = RAW_DIR / "train"
VALID_DIR = RAW_DIR / "valid"
TEST_DIR = RAW_DIR / "test"
REMEDIES_FILE = DATA_DIR / "remedies.json"

# --- models -------------------------------------------------------------
MODELS_DIR = ROOT / "models"
MODEL_FILE = MODELS_DIR / "plant_disease_resnet9.pth"
CLASS_NAMES_FILE = MODELS_DIR / "class_names.json"
HISTORY_FILE = MODELS_DIR / "history.json"

# --- reports ------------------------------------------------------------
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_FILE = REPORTS_DIR / "metrics.json"

# --- uploads ------------------------------------------------------------
UPLOAD_DIR = ROOT / "app" / "static" / "uploads"

# --- dataset facts (verified against the source zip) --------------------
NUM_CLASSES = 38
EXPECTED_TRAIN = 70295
EXPECTED_VALID = 17572
EXPECTED_TEST = 33

# Source zip layout. The archive ships the dataset TWICE in two folders whose
# names differ only in case; on case-insensitive Windows the second copy
# collides with the first, which is what makes a naive unzip take hours.
ZIP_CANONICAL_PREFIX = (
    "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/"
)
ZIP_DUPLICATE_PREFIX = "new plant diseases dataset(augmented)/"
ZIP_TEST_PREFIX = "test/"
DEFAULT_ZIP = Path("C:/Users/User/Downloads/archive (1).zip")

# --- training defaults (paper-faithful; all CLI-overridable) ------------
IMG_SIZE = 256          # paper resolution; drop to 128 or 64 on CPU
BATCH_SIZE = 32
EPOCHS = 5              # ResNet9 converges by epoch 2-3; paper states 20
MAX_LR = 0.01
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 0.1
SEED = 42

# Confidence below this triggers the "retake the photo" advisory, matching the
# paper's error analysis (low light / occluded leaves).
LOW_CONFIDENCE_THRESHOLD = 0.60

ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
