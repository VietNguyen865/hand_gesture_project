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

