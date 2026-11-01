"""Simple helpers for videos, frames, labels, and participant-based splits."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import cv2
import numpy as np
import pandas as pd

if __package__:
    from . import config
else:
    import config


PathLike = Union[str, Path]
LABEL_COLUMNS = (
    "image_path", "source_video", "gesture", "participant_id", "session_id",
    "view_angle", "split",
)


def ensure_directories(directories: Iterable[PathLike]) -> None:
    """Create the requested directories, including their parents."""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def scan_videos(
    video_dir: PathLike,
    extensions: Sequence[str] = config.VIDEO_EXTENSIONS,
) -> List[Path]:
    """Find videos recursively, including uppercase extensions such as .MP4."""
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    extensions = {extension.lower() for extension in extensions}
    return sorted(
        path for path in video_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def get_video_info(video_path: PathLike) -> Dict[str, float]:
    """Read FPS, frame count, dimensions, and duration (0 if FPS is unknown)."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "duration": frame_count / fps if fps > 0 else 0.0,
        }
    finally:
        capture.release()


def extract_frames(
    video_path: PathLike,
    output_dir: PathLike,
    participant_id: str,
    session_id: str,
    gesture: str,
    view_angle: Optional[str] = None,
    sample_interval: int = config.FRAME_SAMPLE_INTERVAL,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_frames: Optional[int] = config.MAX_FRAMES_PER_VIDEO,
) -> List[Path]:
    """Extract frames into output_dir/<video_stem>/ without overwriting.

    Frame numbers are zero-based. start_frame is included; end_frame is
    excluded. Use distinct video names for different recordings. An existing
    nonempty output folder raises an error rather than mixing recordings.
    view_angle may be None for the other class (no command).
    """
    if gesture not in config.GESTURE_CLASSES:
        raise ValueError(f"Unknown gesture: {gesture}")
    if view_angle not in (*config.VIEW_ANGLES, None):
        raise ValueError(f"Unknown view angle: {view_angle}")
    if view_angle is None and gesture != "other":
        raise ValueError("view_angle is required for command gestures.")
    if sample_interval < 1 or start_frame < 0:
        raise ValueError("sample_interval must be positive; start_frame must be >= 0.")
    if end_frame is not None and end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive or None.")

    video_path = Path(video_path)
    frame_dir = Path(output_dir) / video_path.stem
    if frame_dir.exists() and any(frame_dir.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {frame_dir}")

    capture = cv2.VideoCapture(str(video_path))
    saved_paths = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        frame_dir.mkdir(parents=True, exist_ok=True)
        angle_part = f"_{view_angle}" if view_angle is not None else ""
        prefix = f"{participant_id}_{session_id}_{gesture}{angle_part}"
        frame_number = 0
        while end_frame is None or frame_number < end_frame:
            success, frame = capture.read()
            if not success:
                break
            if frame_number >= start_frame and (frame_number - start_frame) % sample_interval == 0:
                image_path = frame_dir / f"{prefix}_{frame_number:06d}.jpg"
                success, encoded = cv2.imencode(".jpg", frame)
                if not success:
                    raise OSError(f"Cannot encode frame: {image_path}")
                # Binary file access also supports Vietnamese paths on Windows.
                with image_path.open("xb") as image_file:
                    image_file.write(encoded.tobytes())
                saved_paths.append(image_path)
                if max_frames is not None and len(saved_paths) >= max_frames:
                    break
            frame_number += 1
    finally:
        capture.release()
    return saved_paths


def _read_image(image_path: PathLike) -> Optional[np.ndarray]:
    """Read an image with a Unicode path; return None for unreadable files."""
    try:
        data = np.frombuffer(Path(image_path).read_bytes(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, cv2.error):
        return None


def compute_frame_similarity(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    method: str = "cosine",
) -> float:
    """Return cosine similarity in [0, 1] after grayscale conversion/resize.

    This is a simple duplicate filter. Tune its threshold on example frames;
    a large static background can hide small changes in the hand.
    """
    if method != "cosine":
        raise ValueError("Only method='cosine' is supported.")
    vectors = []
    for frame in (frame_a, frame_b):
        if frame is None or frame.size == 0:
            raise ValueError("Cannot compare empty frames.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        vectors.append(cv2.resize(gray, (64, 64)).astype(np.float32).ravel() / 255.0)
    a, b = vectors
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return float(np.array_equal(a, b))
    return float(np.clip(np.dot(a, b) / denominator, 0.0, 1.0))


def remove_duplicate_frames(
    frame_paths: Sequence[PathLike],
    similarity_threshold: float = 0.98,
    method: str = "cosine",
) -> List[Path]:
    """Keep distinct readable frames in input order; never delete files.

    Call separately for each video. Compare with the last retained frame.
    """
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1.")
    if method != "cosine":
        raise ValueError("Only method='cosine' is supported.")
    kept_paths = []
    previous_frame = None
    for frame_path in frame_paths:
        frame = _read_image(frame_path)
        if frame is None:
            continue
        if previous_frame is None or compute_frame_similarity(previous_frame, frame, method) < similarity_threshold:
            kept_paths.append(Path(frame_path))
            previous_frame = frame
    return kept_paths


def check_frame_quality(
    frame: Optional[np.ndarray],
    min_width: int = 100,
    min_height: int = 100,
    blur_threshold: Optional[float] = None,
) -> bool:
    """Check image dimensions and optionally Laplacian variance for blur."""
    if frame is None or frame.size == 0 or frame.ndim not in (2, 3):
        return False
    height, width = frame.shape[:2]
    if width < min_width or height < min_height:
        return False
    if blur_threshold is not None:
        if blur_threshold < 0:
            raise ValueError("blur_threshold must be nonnegative.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return bool(cv2.Laplacian(gray, cv2.CV_64F).var() >= blur_threshold)
    return True


def clean_frame_paths(
    frame_paths: Sequence[PathLike],
    remove_blurry: bool = False,
    remove_duplicates: bool = True,
    blur_threshold: Optional[float] = None,
    similarity_threshold: float = 0.98,
) -> List[Path]:
    """Filter one video's frames without deleting their files.

    Supply blur_threshold when enabling remove_blurry. Always reject
    unreadable images and images smaller than 100 x 100 pixels.
    """
    if remove_blurry and blur_threshold is None:
        raise ValueError("Provide blur_threshold when remove_blurry=True.")
    threshold = blur_threshold if remove_blurry else None
    paths = [Path(path) for path in frame_paths if check_frame_quality(_read_image(path), blur_threshold=threshold)]
    if remove_duplicates:
        paths = remove_duplicate_frames(
            paths,
            similarity_threshold=similarity_threshold,
            method="cosine",
        )
    return paths


def create_label_record(
    image_path: PathLike,
    source_video: PathLike,
    gesture: str,
    participant_id: str,
    session_id: str,
    view_angle: Optional[str] = None,
    split: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Build a record with paths relative to PROJECT_ROOT, using '/' separators.

    Relative input paths are also interpreted from PROJECT_ROOT.
    For the other class, view_angle may be None (an empty CSV cell).
    """
    root = config.PROJECT_ROOT.resolve()
    record = {
        "image_path": (root / image_path).resolve().relative_to(root).as_posix(),
        "source_video": (root / source_video).resolve().relative_to(root).as_posix(),
        "gesture": gesture,
        "participant_id": participant_id,
        "session_id": session_id,
        "view_angle": view_angle,
        "split": split,
    }
    _validate_labels(pd.DataFrame([record]))
    return record


def _validate_labels(labels: pd.DataFrame) -> None:
    """Check required fields and the class/angle/split values."""
    missing = set(LABEL_COLUMNS) - set(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    for column in ("image_path", "source_video", "gesture", "participant_id", "session_id"):
        if labels[column].isna().any() or labels[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Empty values in required column: {column}")
    if not labels["gesture"].isin(config.GESTURE_CLASSES).all():
        raise ValueError("Unknown gesture in labels.")
    angles = labels["view_angle"].fillna("")
    if not angles.isin((*config.VIEW_ANGLES, "")).all():
        raise ValueError("view_angle must be front, left, right, or empty.")
    if (angles.eq("") & labels["gesture"].ne("other")).any():
        raise ValueError("view_angle is required for command gestures.")
    if not labels["split"].fillna("").isin(("train", "validation", "test", "")).all():
        raise ValueError("split must be train, validation, test, or empty.")
    if labels["image_path"].duplicated().any():
        raise ValueError("Duplicate image_path values in labels.")


def save_labels(
    records: Sequence[Dict[str, Optional[str]]],
    output_csv: PathLike,
) -> None:
    """Validate records and save CSV; None becomes an empty CSV cell."""
    labels = pd.DataFrame(records) if len(records) else pd.DataFrame(columns=LABEL_COLUMNS)
    _validate_labels(labels)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    labels.loc[:, list(LABEL_COLUMNS)].to_csv(output_csv, index=False, encoding="utf-8-sig")


def load_labels(labels_csv: PathLike) -> pd.DataFrame:
    """Read CSV as strings, preserving participant IDs and empty cells as None."""
    labels = pd.read_csv(labels_csv, dtype=str, keep_default_na=False).replace({"": None})
    _validate_labels(labels)
    return labels


def update_split_column(
    labels: pd.DataFrame,
    train_participants: Sequence[str],
    validation_participants: Sequence[str],
    test_participants: Sequence[str],
) -> pd.DataFrame:
    """Return a copy with participant-based splits; reject overlaps/omissions."""
    _validate_labels(labels)
    train = set(train_participants)
    validation = set(validation_participants)
    test = set(test_participants)
    if train & validation or train & test or validation & test:
        raise ValueError("Participant groups must not overlap.")
    assignments = {participant: "train" for participant in train}
    assignments.update({participant: "validation" for participant in validation})
    assignments.update({participant: "test" for participant in test})
    result = labels.copy()
    result["split"] = result["participant_id"].map(assignments)
    if result["split"].isna().any():
        missing = result.loc[result["split"].isna(), "participant_id"].unique().tolist()
        raise ValueError(f"Participants without a split: {missing}")
    return result


def split_by_participant(
    labels: pd.DataFrame,
    train_participants: Sequence[str],
    validation_participants: Sequence[str],
    test_participants: Sequence[str],
) -> pd.DataFrame:
    """Convenience wrapper; validation is handled by update_split_column."""
    return update_split_column(labels, train_participants, validation_participants, test_participants)
