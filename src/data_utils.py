"""Utilities for collecting, validating, labeling, and splitting data.

This module should only handle data management. Image enhancement,
segmentation, feature extraction, and model training belong in their
respective modules.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import cv2
import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def ensure_directories(directories: Iterable[PathLike]) -> None:
    """Create data/output directories if they do not already exist.

    Parameters
    ----------
    directories:
        Directory paths such as the frame, mask, crop, or result folders.
    """
    pass


def scan_videos(
    video_dir: PathLike,
    extensions: Sequence[str] = (".mp4", ".avi", ".mov"),
) -> List[Path]:
    """Recursively find video files in a directory.

    Parameters
    ----------
    video_dir:
        Root directory containing participant/session folders.
    extensions:
        Accepted video extensions.

    Returns
    -------
    list[Path]
        Sorted paths of all matching video files.
    """
    pass


def get_video_info(video_path: PathLike) -> Dict[str, float]:
    """Read basic metadata from one video.

    The returned dictionary should contain FPS, frame count, width, height,
    and duration in seconds. Raise an error if the video cannot be opened.
    """
    pass


def extract_frames(
    video_path: PathLike,
    output_dir: PathLike,
    participant_id: str,
    session_id: str,
    gesture: str,
    view_angle: Optional[str] = None,
    sample_interval: int = 5,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> List[Path]:
    """Extract selected frames from one labeled video.

    Save frames using a consistent name such as:
    ``p01_s01_thumbs_up_front_000001.jpg``.

    Parameters
    ----------
    video_path:
        Path to the original video in ``data/raw/videos``.
    output_dir:
        Destination directory for extracted frames.
    participant_id, session_id, gesture, view_angle:
        Metadata used in the output filename and labels.csv.
    sample_interval:
        Save every N-th frame to avoid near-duplicate frames.
    start_frame, end_frame:
        Optional frame range to process.

    Returns
    -------
    list[Path]
        Paths of the frames successfully written to disk.
    """
    pass


def compute_frame_similarity(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    method: str = "cosine",
) -> float:
    """Compute similarity between two frames.

    Frames should be converted to the same grayscale, resized, normalized
    representation before comparison. The returned value should be higher
    when the two frames are more similar.
    """
    pass


def remove_duplicate_frames(
    frame_paths: Sequence[PathLike],
    similarity_threshold: float = 0.98,
    method: str = "cosine",
) -> List[Path]:
    """Remove frames that are almost identical to a previously kept frame.

    This function should read the frame paths in order, compare each frame
    with the last retained frame, and return only the retained paths.
    It should not delete files from disk.
    """
    pass


def check_frame_quality(
    frame: np.ndarray,
    min_width: int = 100,
    min_height: int = 100,
    blur_threshold: Optional[float] = None,
) -> bool:
    """Check whether one frame is usable.

    The check may include image existence, minimum dimensions, and blur
    detection using Laplacian variance when ``blur_threshold`` is provided.
    """
    pass


def clean_frame_paths(
    frame_paths: Sequence[PathLike],
    remove_blurry: bool = True,
    remove_duplicates: bool = True,
    blur_threshold: Optional[float] = None,
    similarity_threshold: float = 0.98,
) -> List[Path]:
    """Filter frame paths using quality and duplicate checks.

    This function should combine ``check_frame_quality`` and
    ``remove_duplicate_frames``. It should return selected paths without
    deleting the original files.
    """
    pass


def create_label_record(
    image_path: PathLike,
    source_video: PathLike,
    gesture: str,
    participant_id: str,
    session_id: str,
    view_angle: str,
    lighting: str,
    background: str,
    hand_side: str,
    split: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Create one metadata record for ``labels.csv``.

    Paths should be stored as relative paths. ``gesture`` must be one of
    ``thumbs_up``, ``thumbs_down``, ``stop_palm``, ``ok``, or ``neutral``.
    """
    pass


def save_labels(
    records: Sequence[Dict[str, Optional[str]]],
    output_csv: PathLike,
) -> None:
    """Save label records as a CSV file and create its parent directory."""
    pass


def load_labels(labels_csv: PathLike) -> pd.DataFrame:
    """Load ``labels.csv`` and validate its required columns."""
    pass


def update_split_column(
    labels: pd.DataFrame,
    train_participants: Sequence[str],
    validation_participants: Sequence[str],
    test_participants: Sequence[str],
) -> pd.DataFrame:
    """Assign train/validation/test based on participant IDs.

    All frames from the same participant must remain in the same split to
    reduce data leakage.
    """
    pass


def split_by_participant(
    labels: pd.DataFrame,
    train_participants: Sequence[str],
    validation_participants: Sequence[str],
    test_participants: Sequence[str],
) -> pd.DataFrame:
    """Create a participant-wise dataset split.

    This public helper should validate that participant groups do not overlap
    and then call ``update_split_column``.
    """
    pass

