import cv2
import numpy as np


if __package__:
    from . import config
else:
    import config

def resize_image(image, width=config.RESIZE_WIDTH, height=config.RESIZE_HEIGHT):
    return cv2.resize(image, (width, height))


def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]


def rotate_image(image, angle, scale=1.0):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale,)
    return cv2.warpAffine(image, matrix, (width, height),)


def adaptive_gamma_hist(image, target=config.GAMMA_TARGET, gamma_min=config.GAMMA_MIN, gamma_max=config.GAMMA_MAX, return_statistics=False,):
    if image is None:
        raise ValueError("Ảnh đầu vào không được là None.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image phải là ảnh màu BGR có dạng H x W x 3.")
    if not 0 < target < 1:
        raise ValueError("target phải nằm trong khoảng (0, 1).")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]
    histogram = cv2.calcHist([value_channel], [0], None, [256], [0, 256],).ravel()

    histogram = histogram / (histogram.sum() + 1e-8)
    cdf = np.cumsum(histogram)

    p10 = np.searchsorted(cdf, 0.10) / 255.0
    p50 = np.searchsorted(cdf, 0.50) / 255.0
    p90 = np.searchsorted(cdf, 0.90) / 255.0

    dark_ratio = histogram[:51].sum()     
    bright_ratio = histogram[205:].sum()

    median_for_gamma = np.clip(p50, 1e-6, 1.0 - 1e-6)

    gamma = np.log(target) / np.log(median_for_gamma)
    gamma = float(np.clip(gamma, gamma_min, gamma_max))

    lookup_table = np.array([((pixel / 255.0) ** gamma) * 255 for pixel in range(256)], dtype=np.uint8,)

    hsv[:, :, 2] = cv2.LUT(value_channel, lookup_table)
    corrected_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if return_statistics:
        statistics = {
            "p10": float(p10),
            "median": float(p50),
            "p90": float(p90),
            "dark_ratio": float(dark_ratio),
            "bright_ratio": float(bright_ratio),
            "contrast_range": float(p90 - p10),
            "gamma": gamma,
        }

        return corrected_image, statistics
    return corrected_image, gamma