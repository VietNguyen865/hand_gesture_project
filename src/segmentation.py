"""Segmentation and mask post-processing utilities.

This module contains classical computer-vision methods for extracting the
hand region from an image.  The intended workflow is:

    color/edge segmentation -> morphology -> connected components
    -> contour -> bounding box -> hand crop

The functions are intentionally kept independent so that the experiments can
compare HSV, Lab/K-means, and edge-based segmentation fairly.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


if __package__:
    from . import config
else:
    import config


BoundingBox = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Region/color-based segmentation
# ---------------------------------------------------------------------------

def segment_hsv(
    image: np.ndarray,
    lower_hsv: Sequence[int] = (0, 20, 40),
    upper_hsv: Sequence[int] = (35, 255, 255),
) -> np.ndarray:
    """Create a binary skin-region mask using HSV thresholds.

    Parameters
    ----------
    image:
        Color image in BGR order.
    lower_hsv, upper_hsv:
        Inclusive lower and upper HSV thresholds.

    Returns
    -------
    np.ndarray
        Binary mask with values 0 or 255.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def segment_lab_kmeans(
    image,
    n_clusters=config.KMEANS_CLUSTERS,
    skin_cluster_indices=None,
    random_state=config.RANDOM_SEED,
    skin_reference_lab=None,
):
    """Tạo mask bằng K-means trên ảnh BGR uint8.

    Chọn cluster bằng chỉ số truyền vào hoặc màu da tham chiếu Lab.
    skin_reference_lab phải lấy từ ảnh uint8 sau cùng preprocessing.
    """
    height, width = image.shape[:2]

    if skin_cluster_indices is not None:
        selected_clusters = list(skin_cluster_indices)

        for cluster_index in selected_clusters:
            if (
                not isinstance(cluster_index, (int, np.integer))
                or not 0 <= cluster_index < n_clusters
            ):
                raise ValueError("Cluster index không hợp lệ.")

    else:
        if skin_reference_lab is None:
            raise ValueError(
                "Cần truyền skin_reference_lab "
                "hoặc skin_cluster_indices."
            )

        reference = np.asarray(skin_reference_lab, dtype=np.float32)
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            raise ValueError(
                "skin_reference_lab phải gồm 3 giá trị hữu hạn [L, a, b]."
            )

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    samples = lab.reshape(-1, 3).astype(np.float32)

    cv2.setRNGSeed(random_state)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2,)
    _, labels, centers = cv2.kmeans(samples, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    if skin_cluster_indices is None:
        distances = np.linalg.norm(centers[:, 1:3] - reference[1:3], axis=1)
        selected_clusters = [int(np.argmin(distances))]

    mask = np.isin(labels.ravel(), selected_clusters)
    return mask.reshape(height, width).astype(np.uint8) * 255



def compare_masks(mask_a, mask_b):
    """So sánh hai binary mask bằng các thống kê không giám sát."""

    # Chuyển mask về dạng boolean
    binary_a = mask_a > 0
    binary_b = mask_b > 0

    total_pixels = binary_a.size

    area_a = np.count_nonzero(binary_a)
    area_b = np.count_nonzero(binary_b)

    intersection = np.count_nonzero(np.logical_and(binary_a, binary_b))
    union = np.count_nonzero(np.logical_or(binary_a, binary_b))
    difference = np.count_nonzero(np.logical_xor(binary_a, binary_b))

    # Tỷ lệ vùng foreground
    foreground_ratio_a = area_a / total_pixels
    foreground_ratio_b = area_b / total_pixels

    # Mức độ giống nhau giữa hai mask
    if union == 0:
        overlap_ratio = 1.0
    else:
        overlap_ratio = intersection / union

    # Tỷ lệ pixel khác nhau
    difference_ratio = difference / total_pixels

    # Đếm connected components, bỏ background label 0
    num_components_a, _ = cv2.connectedComponents(binary_a.astype(np.uint8), connectivity=8)
    num_components_b, _ = cv2.connectedComponents(binary_b.astype(np.uint8), connectivity=8)

    return {
        "foreground_ratio_a": float(foreground_ratio_a),
        "foreground_ratio_b": float(foreground_ratio_b),
        "overlap_ratio": float(overlap_ratio),
        "difference_ratio": float(difference_ratio),
        "area_difference_ratio": float(
            abs(area_a - area_b) / total_pixels
        ),
        "components_a": float(num_components_a - 1),
        "components_b": float(num_components_b - 1),
    }


# ---------------------------------------------------------------------------
# Morphological post-processing
# ---------------------------------------------------------------------------

def apply_morphology(mask, operation="opening_closing", kernel_shape=cv2.MORPH_ELLIPSE, 
                     kernel_size=config.MORPH_KERNEL_SIZE, iterations=config.MORPH_ITERATIONS,):
    if mask is None:
        raise ValueError("mask không được là None.")
    if mask.ndim != 2:
        raise ValueError("mask phải là ảnh grayscale 2 chiều.")
    if iterations <= 0:
        raise ValueError("iterations phải lớn hơn 0.")
    
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    kernel = create_structuring_element(shape=kernel_shape, kernel_size=kernel_size)
    operation = operation.lower()

    if operation == "erosion":
        result = cv2.erode(binary_mask, kernel, iterations=iterations)
    elif operation == "dilation":
        result = cv2.dilate(binary_mask, kernel, iterations=iterations)
    elif operation == "opening":
        result = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "closing":
        result = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == "opening_closing":
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        raise ValueError(
            "operation phải là: "
            "'erosion', 'dilation', 'opening', "
            "'closing' hoặc 'opening_closing'."
        )
    return result


def create_structuring_element(shape=cv2.MORPH_ELLIPSE, kernel_size=config.MORPH_KERNEL_SIZE,):
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size phải là số lẻ và lớn hơn 0.")
    
    return cv2.getStructuringElement(shape, (kernel_size, kernel_size))


# ---------------------------------------------------------------------------
# Connected components and contour-based localization
# ---------------------------------------------------------------------------

def remove_small_components(mask, min_area=config.MIN_CONTOUR_AREA, keep_largest=False,):
    """Remove small connected components from a binary mask.

    Parameters
    ----------
    mask:
        Binary mask with foreground value 255 or any non-zero value.
    min_area:
        Minimum component area to keep.
    keep_largest:
        If True, keep only the largest valid component.
    """
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    valid_components = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_components.append((label, area))

    if keep_largest and valid_components:
        largest_label, _ = max(valid_components, key=lambda item: item[1],)
        valid_components = [(largest_label, 0)]

    clean = np.zeros_like(binary_mask)
    for label, _ in valid_components:
        clean[labels == label] = 255
    return clean


def extract_external_contours(mask, min_area=config.MIN_CONTOUR_AREA) -> List[np.ndarray]:
    """Find and filter external contours from a binary mask."""
    binary_mask = np.where(mask > 0, 255, 0,).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    contours.sort(key=cv2.contourArea,reverse=True,)
    return contours


def select_hand_contour(contours, strategy="largest_area"):
    """Select the contour most likely to represent the hand region."""
    if strategy != "largest_area":
        raise ValueError("strategy hiện chỉ hỗ trợ 'largest_area'.")
    if contours is None or len(contours) == 0:
        return None

    valid_contours = [
        contour
        for contour in contours
        if contour is not None
        and contour.size > 0
        and cv2.contourArea(contour) > 0
    ]
    if not valid_contours:
        return None
    return max(valid_contours, key=cv2.contourArea)


def get_bounding_box(contour, image_shape=None, padding=0):
    """Return a padded and image-clipped ``(x, y, width, height)`` box."""
    if contour is None or contour.size == 0:
        raise ValueError("contour bị rỗng.")

    if not isinstance(padding, int) or padding < 0:
        raise ValueError("padding phải là số nguyên không âm.")

    x, y, width, height = cv2.boundingRect(contour)

    x1 = x - padding
    y1 = y - padding
    x2 = x + width + padding
    y2 = y + height + padding

    if image_shape is not None:
        if len(image_shape) < 2:
            raise ValueError("image_shape phải chứa chiều cao và chiều rộng.")

        image_height, image_width = image_shape[:2]
        if image_height <= 0 or image_width <= 0:
            raise ValueError("Kích thước ảnh phải lớn hơn 0.")
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box không chứa vùng ảnh hợp lệ.")
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def crop_hand(image, bounding_box):
    """Crop the hand region from an image using a bounding box."""
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào bị rỗng.")
    if image.ndim not in (2, 3):
        raise ValueError("Ảnh phải có 2 hoặc 3 chiều.")
    if len(bounding_box) != 4:
        raise ValueError("bounding_box phải gồm (x, y, width, height).")
    if not all(
        isinstance(value, (int, np.integer))
        for value in bounding_box
    ):
        raise ValueError("Các giá trị bounding_box phải là số nguyên.")

    x, y, width, height = bounding_box

    if width <= 0 or height <= 0:
        raise ValueError("width và height phải lớn hơn 0.")

    image_height, image_width = image.shape[:2]

    # Giới hạn tọa độ, tránh chỉ số âm khi cắt bằng NumPy
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_width, x + width)
    y2 = min(image_height, y + height)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box nằm ngoài ảnh.")

    return image[y1:y2, x1:x2].copy()


def contour_to_mask(image_shape, contour):
    """Fill one contour to create a binary object mask."""
    if len(image_shape) < 2:
        raise ValueError("image_shape phải chứa chiều cao và chiều rộng.")

    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Kích thước ảnh phải lớn hơn 0.")
    mask = np.zeros((height, width), dtype=np.uint8)
    if contour is None or contour.size == 0:
        return mask
    cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED)

    return mask


# ---------------------------------------------------------------------------
# Edge-based segmentation
# ---------------------------------------------------------------------------

def compute_gradient_magnitude(gray, kernel_size=3):
    """Compute Sobel gradient magnitude from a grayscale image."""
    gx = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0, ksize=kernel_size)
    gy = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=kernel_size)
    magnitude = cv2.magnitude(gx, gy)
    return cv2.normalize(magnitude, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def canny_edge_map(gray, low_threshold=80, high_threshold=160, blur_kernel_size=5):
    """Create a binary Canny edge map."""
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), sigmaX=0)
    return cv2.Canny(blurred, threshold1=low_threshold, threshold2=high_threshold, L2gradient=True)


def link_edges_with_closing(edge_map, kernel_size=config.MORPH_KERNEL_SIZE):
    """Connect small gaps in an edge map using morphological closing."""
    binary = np.where(edge_map > 0, 255, 0).astype(np.uint8)
    kernel = create_structuring_element(shape=cv2.MORPH_ELLIPSE, kernel_size=kernel_size)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)


def filter_small_edge_components(edge_map, min_area=30):
    """Remove small connected components from an edge map."""
    return remove_small_components(edge_map, min_area=min_area, keep_largest=False)


def edge_based_segmentation(image, low_threshold=80, high_threshold=160, min_area=config.MIN_CONTOUR_AREA, 
                            blur_kernel_size=3, closing_kernel_size=3, min_edge_area=30):
    """Run the complete edge-to-mask segmentation pipeline.

    Expected stages are grayscale conversion, smoothing, Canny/Sobel edge
    detection, gap closing, component filtering, contour extraction, and
    contour filling.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient = compute_gradient_magnitude(gray)

    edges = canny_edge_map(
        gray,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        blur_kernel_size=blur_kernel_size,
    )

    linked_edges = link_edges_with_closing(edges, kernel_size=closing_kernel_size)
    clean_edges = filter_small_edge_components(linked_edges, min_area=min_edge_area)
    contours = extract_external_contours(clean_edges, min_area=min_area)
    mask = np.zeros_like(gray)

    if contours:
        cv2.drawContours(mask, contours, contourIdx=-1, color=255, thickness=cv2.FILLED)

    colored = image.copy()
    colored[mask > 0] = (0, 200, 0)
    overlay = cv2.addWeighted(image, 0.7, colored, 0.3, 0)

    cv2.drawContours(overlay, contours, contourIdx=-1, color=(0, 255, 255), thickness=1)

    return {
        "gray": gray,
        "gradient": gradient,
        "edges": edges,
        "linked_edges": linked_edges,
        "clean_edges": clean_edges,
        "mask": mask,
        "overlay": overlay,
    }

