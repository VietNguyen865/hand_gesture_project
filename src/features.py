"""Trích xuất đặc trưng HOG, Canny và gamma-Otsu của Người 3."""

from typing import Tuple

import cv2
import numpy as np
from skimage.feature import hog


DEFAULT_HOG_SIZE = (128, 128)
DEFAULT_HOG_CELL_SIZE = (8, 8)
DEFAULT_HOG_CELLS_PER_BLOCK = (2, 2)
DEFAULT_HOG_ORIENTATIONS = 9

SHAPE_FEATURE_NAMES = (
    "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6", "hu_7",
    "area_ratio", "perimeter_ratio", "aspect_ratio", "extent", "solidity",
    "centroid_x", "centroid_y",
)


def to_grayscale(image: np.ndarray, color_order: str = "BGR") -> np.ndarray:
    """Đổi ảnh BGR/RGB uint8 sang grayscale; giữ nguyên nếu ảnh đã là grayscale."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array.")
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixel values.")
    if image.ndim == 2:
        return image
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape HxW or HxWx3.")

    conversions = {"BGR": cv2.COLOR_BGR2GRAY, "RGB": cv2.COLOR_RGB2GRAY}
    order = color_order.upper()
    if order not in conversions:
        raise ValueError("color_order must be 'BGR' or 'RGB'.")
    return cv2.cvtColor(image, conversions[order])


def resize_grayscale_preserving_aspect(
    gray: np.ndarray,
    size: Tuple[int, int] = DEFAULT_HOG_SIZE,
) -> np.ndarray:
    """Giữ tỷ lệ ảnh trong kích thước ``(rộng, cao)`` và đệm phần còn trống ở giữa."""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.size == 0:
        raise ValueError("gray must be a non-empty 2D NumPy array.")
    if gray.dtype != np.uint8:
        raise ValueError("gray must use uint8 pixel values.")
    if len(size) != 2 or any(not isinstance(value, (int, np.integer)) or value <= 0 for value in size):
        raise ValueError("size must contain positive integer (width, height).")

    target_width, target_height = map(int, size)
    source_height, source_width = gray.shape
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, min(target_width, round(source_width * scale)))
    resized_height = max(1, min(target_height, round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (resized_width, resized_height), interpolation=interpolation)

    canvas = np.zeros((target_height, target_width), dtype=np.uint8)
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return canvas


def extract_hog_features(
    image: np.ndarray,
    image_size: Tuple[int, int] = DEFAULT_HOG_SIZE,
    cell_size: Tuple[int, int] = DEFAULT_HOG_CELL_SIZE,
    cells_per_block: Tuple[int, int] = DEFAULT_HOG_CELLS_PER_BLOCK,
    orientations: int = DEFAULT_HOG_ORIENTATIONS,
    block_norm: str = "L2-Hys",
    color_order: str = "BGR",
) -> np.ndarray:
    """Trích HOG bằng cùng các tham số cell, block và bins trong bài lab.

    Mỗi phiếu histogram được cân theo độ lớn gradient. Chuẩn hóa block giúp
    giảm ảnh hưởng của thay đổi độ sáng/tương phản cục bộ.
    """
    gray = resize_grayscale_preserving_aspect(to_grayscale(image, color_order), image_size)
    if len(cell_size) != 2 or any(not isinstance(value, (int, np.integer)) or value <= 0 for value in cell_size):
        raise ValueError("cell_size must contain two positive integers.")
    if len(cells_per_block) != 2 or any(not isinstance(value, (int, np.integer)) or value <= 0 for value in cells_per_block):
        raise ValueError("cells_per_block must contain two positive integers.")
    if not isinstance(orientations, int) or orientations <= 0:
        raise ValueError("orientations must be a positive integer.")
    if block_norm not in {"L1", "L1-sqrt", "L2", "L2-Hys"}:
        raise ValueError("block_norm must be L1, L1-sqrt, L2, or L2-Hys.")

    target_width, target_height = map(int, image_size)
    cell_width, cell_height = map(int, cell_size)
    block_width = cell_width * int(cells_per_block[1])
    block_height = cell_height * int(cells_per_block[0])
    if (
        block_width > target_width
        or block_height > target_height
        or target_width % cell_width
        or target_height % cell_height
    ):
        raise ValueError("image_size must contain whole cells and fit at least one HOG block.")

    features = hog(
        gray.astype(np.float32) / 255.0,
        orientations=orientations,
        pixels_per_cell=(cell_height, cell_width),
        cells_per_block=tuple(map(int, cells_per_block)),
        block_norm=block_norm,
        visualize=False,
        transform_sqrt=False,
        feature_vector=True,
        channel_axis=None,
    )
    return np.asarray(features, dtype=np.float32)


def extract_shape_features(mask: np.ndarray) -> np.ndarray:
    """Trả về 7 moment Hu dạng log và 7 đặc trưng hình học đã chuẩn hóa."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
        raise ValueError("mask must be a non-empty 2D NumPy array.")
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 0]
    if not contours:
        raise ValueError("No non-empty contour found in mask.")

    contour = max(contours, key=cv2.contourArea)
    height, width = binary.shape
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, box_width, box_height = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        raise ValueError("Contour has zero spatial moment.")

    hu = cv2.HuMoments(moments).reshape(-1)
    hu_log = -np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-30))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    geometry = np.array(
        [
            area / (height * width),
            perimeter / (2.0 * (height + width)),
            box_width / box_height,
            area / (box_width * box_height),
            area / hull_area if hull_area > 0 else 0.0,
            (moments["m10"] / moments["m00"]) / width,
            (moments["m01"] / moments["m00"]) / height,
        ],
        dtype=np.float64,
    )
    return np.concatenate((hu_log, geometry)).astype(np.float32)


def extract_canny_shape_features(
    image: np.ndarray,
    low_threshold: int = 80,
    high_threshold: int = 160,
    **segmentation_options,
) -> np.ndarray:
    """Chạy Canny và trích Hu/hình học; mask rỗng được biểu diễn bằng vector 0."""
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty grayscale or BGR image.")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("color image must have three channels.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if __package__:
        from .segmentation import edge_based_segmentation
    else:
        from segmentation import edge_based_segmentation

    result = edge_based_segmentation(
        image,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        **segmentation_options,
    )
    if cv2.countNonZero(result["mask"]) == 0:
        return np.zeros(len(SHAPE_FEATURE_NAMES), dtype=np.float32)
    return extract_shape_features(result["mask"])


def extract_gamma_shape_features(
    image: np.ndarray,
    gamma: float = 0.8,
    foreground: str = "bright",
    **segmentation_options,
) -> np.ndarray:
    """Trích Hu/hình học sau gamma, Otsu và morphology."""
    gray = to_grayscale(image)
    if __package__:
        from .segmentation import gamma_otsu_segmentation
    else:
        from segmentation import gamma_otsu_segmentation

    result = gamma_otsu_segmentation(
        gray,
        gamma=gamma,
        foreground=foreground,
        **segmentation_options,
    )
    if cv2.countNonZero(result["mask"]) == 0:
        return np.zeros(len(SHAPE_FEATURE_NAMES), dtype=np.float32)
    return extract_shape_features(result["mask"])
