from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_VIDEO_DIR = RAW_DIR / "videos"

PROCESSED_DIR = DATA_DIR / "processed"
FRAME_DIR = PROCESSED_DIR / "frames"
MASK_DIR = PROCESSED_DIR / "masks"
CROP_DIR = PROCESSED_DIR / "crops"
FEATURE_DIR = PROCESSED_DIR / "features"

LABELS_PATH = DATA_DIR / "labels.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"


# ============================================================
# Gesture labels
# ============================================================

GESTURE_CLASSES = (
    "thumbs_up",
    "thumbs_down",
    "stop_palm",
    "ok",
    "other",
)

VIEW_ANGLES = (
    "front",
    "left",
    "right",
)


# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED = 42


# ============================================================
# Input and frame extraction
# ============================================================

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
)

FRAME_SAMPLE_INTERVAL = 5
MAX_FRAMES_PER_VIDEO = None


# ============================================================
# Image preprocessing
# ============================================================

RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480

FILTER_METHOD = "median"
MEDIAN_KERNEL_SIZE = 5
GAUSSIAN_KERNEL_SIZE = 5

GAMMA_METHOD = "adaptive_hist"
GAMMA_TARGET = 0.5
GAMMA_MIN = 0.6
GAMMA_MAX = 1.6


# ============================================================
# Segmentation and morphology
# ============================================================

SEGMENTATION_METHODS = (
    "hsv",
    "lab_kmeans",
)

KMEANS_CLUSTERS = 3

MORPH_KERNEL_SIZE = 5
MORPH_ITERATIONS = 1

MIN_CONTOUR_AREA = 500


# ============================================================
# Feature extraction
# ============================================================

FEATURE_METHODS = (
    "hu_shape",
    "hog",
)


# ============================================================
# Classical recognition
# ============================================================

CLASSIFIER_METHODS = (
    "knn",
    "svm",
)

KNN_NEIGHBORS = 5

SVM_KERNEL = "rbf"
SVM_C = 10.0
SVM_GAMMA = "scale"