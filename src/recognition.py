"""Các bộ phân loại cổ điển dùng trong thí nghiệm nhận dạng cử chỉ."""

from typing import Optional

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if __package__:
    from . import config
else:
    import config


def _validate_features(features: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D feature matrix.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return values


def train_svm(
    features: np.ndarray,
    labels,
    kernel: str = config.SVM_KERNEL,
    c: float = config.SVM_C,
    gamma=config.SVM_GAMMA,
    class_weight: Optional[str] = None,
):
    """Huấn luyện SVM có chuẩn hóa; bộ chuẩn hóa chỉ học từ dữ liệu train."""
    x_train = _validate_features(features, "features")
    y_train = np.asarray(labels)
    if y_train.ndim != 1 or len(y_train) != len(x_train):
        raise ValueError("labels must be a 1D array matching the number of feature rows.")
    if any(
        str(label).strip().casefold() in {"", "nan", "none", "<na>", "nat"}
        for label in y_train.tolist()
    ):
        raise ValueError("labels cannot contain missing or empty values.")
    if np.unique(y_train).size < 2:
        raise ValueError("SVM training requires at least two classes.")
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be a positive finite value.")
    if not isinstance(kernel, str) or kernel.lower() not in {"linear", "poly", "rbf", "sigmoid"}:
        raise ValueError("kernel must be linear, poly, rbf, or sigmoid.")
    if isinstance(gamma, str):
        if gamma not in {"scale", "auto"}:
            raise ValueError("gamma must be 'scale', 'auto', or a positive number.")
    elif isinstance(gamma, (int, float, np.integer, np.floating)):
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("numeric gamma must be positive and finite.")
        gamma = float(gamma)
    else:
        raise ValueError("gamma must be 'scale', 'auto', or a positive number.")

    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel=kernel.lower(),
            C=float(c),
            gamma=gamma,
            class_weight=class_weight,
        ),
    ).fit(x_train, y_train)


def predict_svm(model, features: np.ndarray) -> np.ndarray:
    """Dự đoán nhãn bằng estimator đã huấn luyện từ ``train_svm``."""
    if not hasattr(model, "predict"):
        raise TypeError("model must be a fitted estimator with predict().")
    return model.predict(_validate_features(features, "features"))
