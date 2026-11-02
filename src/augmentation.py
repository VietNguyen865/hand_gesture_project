import cv2
import numpy as np


def _check_image(image):
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image phải là mảng NumPy không rỗng.")

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image phải là ảnh BGR uint8 có dạng H x W x 3.")


def random_rotate(image, angle_range=(-15, 15), rng=None):
    _check_image(image)
    if angle_range[0] > angle_range[1]:
        raise ValueError("Khoảng góc xoay không hợp lệ.")
    rng = np.random.default_rng() if rng is None else rng
    angle = rng.uniform(*angle_range)
    height, width = image.shape[:2]
    center = ((width - 1) / 2, (height - 1) / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def random_translate(image, max_shift=20, rng=None):
    _check_image(image)
    if not isinstance(max_shift, int) or max_shift < 0:
        raise ValueError("max_shift phải là số nguyên không âm.")
    rng = np.random.default_rng() if rng is None else rng
    shift_x, shift_y = rng.integers(
        -max_shift,
        max_shift + 1,
        size=2,
    )
    matrix = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y],
    ])
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def random_scale(image, scale_range=(0.9, 1.1), rng=None):
    _check_image(image)
    if not 0 < scale_range[0] <= scale_range[1]:
        raise ValueError("scale_range phải thỏa mãn 0 < min <= max.")
    rng = np.random.default_rng() if rng is None else rng
    scale = rng.uniform(*scale_range)
    height, width = image.shape[:2]
    center = ((width - 1) / 2, (height - 1) / 2)
    matrix = cv2.getRotationMatrix2D(center, 0, scale)

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def random_shear(image, shear_range=(-0.15, 0.15), rng=None):
    _check_image(image)
    if shear_range[0] > shear_range[1]:
        raise ValueError("Khoảng shear không hợp lệ.")
    rng = np.random.default_rng() if rng is None else rng
    shear = rng.uniform(*shear_range)
    height, width = image.shape[:2]
    matrix = np.float32([
        [1, shear, -shear * (height - 1) / 2],
        [0, 1, 0],
    ])

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def random_flip(image, probability=0.5, rng=None):
    _check_image(image)
    if not 0 <= probability <= 1:
        raise ValueError("probability phải nằm trong khoảng [0, 1].")
    rng = np.random.default_rng() if rng is None else rng
    if rng.random() < probability:
        return cv2.flip(image, 1)
    return image.copy()


def augment_image(image, seed=None):
    _check_image(image)
    rng = np.random.default_rng(seed)
    operations = (
        random_rotate,
        random_translate,
        random_scale,
        random_shear,
    )
    operation = operations[int(rng.integers(len(operations)))]
    augmented = operation(image, rng=rng)
    return random_flip(augmented, probability=0.5, rng=rng,)