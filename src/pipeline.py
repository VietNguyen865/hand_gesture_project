"""Pipeline phân loại HOG, Canny và gamma-Otsu cho Người 3."""

from typing import Dict, Iterable, Optional

import numpy as np

if __package__:
    from .evaluation import evaluate_classification
    from .features import (
        extract_canny_shape_features,
        extract_gamma_shape_features,
        extract_hog_features,
    )
    from .recognition import predict_svm, train_svm
else:
    from evaluation import evaluate_classification
    from features import (
        extract_canny_shape_features,
        extract_gamma_shape_features,
        extract_hog_features,
    )
    from recognition import predict_svm, train_svm


def extract_feature_matrix(
    images: Iterable[np.ndarray],
    method: str,
    hog_options: Optional[Dict] = None,
    canny_options: Optional[Dict] = None,
    gamma_options: Optional[Dict] = None,
) -> np.ndarray:
    """Trích ma trận đặc trưng đồng nhất từ các crop bàn tay đã nạp sẵn."""
    feature_rows = []
    for image in images:
        if method == "hog":
            row = extract_hog_features(image, **(hog_options or {}))
        elif method == "canny_shape":
            row = extract_canny_shape_features(image, **(canny_options or {}))
        elif method == "gamma_shape":
            row = extract_gamma_shape_features(image, **(gamma_options or {}))
        else:
            raise ValueError("method must be 'hog', 'canny_shape', or 'gamma_shape'.")
        feature_rows.append(row)
    if not feature_rows:
        raise ValueError("images must contain at least one hand crop.")
    return np.vstack(feature_rows).astype(np.float32, copy=False)


def run_classification_experiment(
    train_images: Iterable[np.ndarray],
    train_labels,
    evaluation_images: Iterable[np.ndarray],
    evaluation_labels,
    method: str,
    hog_options: Optional[Dict] = None,
    canny_options: Optional[Dict] = None,
    svm_options: Optional[Dict] = None,
    gamma_options: Optional[Dict] = None,
):
    """Huấn luyện một nhánh SVM và đánh giá trên split do người gọi chọn.

    Dùng validation khi chỉnh ngưỡng hoặc tham số mô hình. Chỉ đánh giá trên
    test chính thức sau khi đã cố định các lựa chọn đó.
    """
    train_images = list(train_images)
    evaluation_images = list(evaluation_images)
    train_labels = np.asarray(train_labels)
    evaluation_labels = np.asarray(evaluation_labels)
    if len(train_images) != len(train_labels):
        raise ValueError("train_images and train_labels must have equal length.")
    if len(evaluation_images) != len(evaluation_labels):
        raise ValueError("evaluation_images and evaluation_labels must have equal length.")
    if not evaluation_images:
        raise ValueError("evaluation_images must contain at least one hand crop.")

    x_train = extract_feature_matrix(
        train_images, method, hog_options, canny_options, gamma_options
    )
    x_eval = extract_feature_matrix(
        evaluation_images, method, hog_options, canny_options, gamma_options
    )
    model = train_svm(x_train, train_labels, **(svm_options or {}))
    predictions = predict_svm(model, x_eval)
    metrics = evaluate_classification(evaluation_labels, predictions)
    return {"model": model, "predictions": predictions, "metrics": metrics}


def run_hog_experiment(
    train_images,
    train_labels,
    evaluation_images,
    evaluation_labels,
    hog_options: Optional[Dict] = None,
    svm_options: Optional[Dict] = None,
):
    """Chạy baseline grayscale-HOG cùng SVM."""
    return run_classification_experiment(
        train_images,
        train_labels,
        evaluation_images,
        evaluation_labels,
        method="hog",
        hog_options=hog_options,
        svm_options=svm_options,
    )


def run_canny_experiment(
    train_images,
    train_labels,
    evaluation_images,
    evaluation_labels,
    canny_options: Optional[Dict] = None,
    svm_options: Optional[Dict] = None,
):
    """Chạy nhánh Canny-contour với đặc trưng Hu/hình học và SVM."""
    return run_classification_experiment(
        train_images,
        train_labels,
        evaluation_images,
        evaluation_labels,
        method="canny_shape",
        canny_options=canny_options,
        svm_options=svm_options,
    )


def run_gamma_experiment(
    train_images,
    train_labels,
    evaluation_images,
    evaluation_labels,
    gamma_options: Optional[Dict] = None,
    svm_options: Optional[Dict] = None,
):
    """Chạy gamma-Otsu, contour, Hu/hình học và SVM."""
    return run_classification_experiment(
        train_images,
        train_labels,
        evaluation_images,
        evaluation_labels,
        method="gamma_shape",
        gamma_options=gamma_options,
        svm_options=svm_options,
    )


