"""Đọc manifest MANUS-HaGRID và ảnh crop cục bộ hoặc từ Hugging Face."""

import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


SOURCE_TO_PROJECT_LABEL = {
    "like": "thumbs_up",
    "thumbs_up": "thumbs_up",
    "dislike": "thumbs_down",
    "thumbs_down": "thumbs_down",
    "stop": "stop_palm",
    "palm": "stop_palm",
    "ok": "ok",
    "no_gesture": "other",
}
MAIN_SOURCE_LABELS = frozenset(("like", "thumbs_up", "dislike", "thumbs_down", "stop", "palm", "ok"))


def _find_column(frame: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for name in candidates:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(f"Manifest is missing one of these columns: {list(candidates)}")
    return None


def _includes_modality(value, modality: str) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    if isinstance(value, str):
        return modality.casefold() in value.casefold()
    if isinstance(value, dict):
        value = value.keys()
    return any(modality.casefold() in str(item).casefold() for item in value)


def _assign_project_labels(manifest: pd.DataFrame) -> pd.DataFrame:
    result = manifest.copy()
    labels = result["source_label"].fillna("").astype("string").str.strip().str.casefold()
    result["source_label"] = labels
    result["project_label"] = labels.map(SOURCE_TO_PROJECT_LABEL)
    return result


def _normalize_manifest(
    manifest: pd.DataFrame,
    config: str,
) -> pd.DataFrame:
    config_column = _find_column(manifest, ("config", "dataset_config"), required=False)
    if config_column:
        manifest = manifest.loc[manifest[config_column].astype(str).str.casefold() == config.casefold()].copy()

    columns = {
        "sample_id": _find_column(manifest, ("sample_id", "manus_sample_id")),
        "source_id": _find_column(manifest, ("source_image_id", "source_id", "image_id"), required=False),
        "source_label": _find_column(manifest, ("gesture_label", "source_label", "label", "gesture")),
        "split": _find_column(manifest, ("split", "source_split")),
        "shard": _find_column(manifest, ("shard", "shard_path", "relative_shard_path", "tar_path")),
        "available": _find_column(manifest, ("available_modalities", "available_files")),
    }

    normalized = manifest.copy()
    normalized["sample_id"] = manifest[columns["sample_id"]].astype("string").str.strip()
    source_column = columns["source_id"]
    normalized["source_id"] = (
        manifest[source_column].astype("string").str.strip()
        if source_column
        else normalized["sample_id"]
    )
    normalized["source_label"] = manifest[columns["source_label"]].astype("string").str.strip().str.casefold()
    normalized["split"] = manifest[columns["split"]].astype("string").str.strip().str.casefold()
    normalized["shard"] = manifest[columns["shard"]].astype("string").str.strip()
    modalities = manifest[columns["available"]]
    normalized["has_hand_rgb"] = modalities.map(lambda value: _includes_modality(value, "hand_rgb"))
    normalized = _assign_project_labels(normalized)

    required_values = ("sample_id", "source_id", "split", "shard")
    for column in required_values:
        if normalized[column].isna().any() or normalized[column].eq("").any():
            raise ValueError(f"Manifest contains missing values in {column!r}.")
    if normalized["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample IDs.")
    if not normalized["split"].isin(("train", "test")).all():
        raise ValueError("Expected official train/test split values in the manifest.")
    return normalized


def _join_prompt_labels(
    manifest: pd.DataFrame,
    prompt_path: Path,
) -> pd.DataFrame:
    """Nối nhãn train từ prompt bằng tên ảnh nguồn."""
    if not prompt_path.is_file():
        return manifest

    prompt_labels = {}
    with prompt_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = str(record.get("image") or "").replace("\\", "/")
            source_id = PurePosixPath(image_path).stem
            label = str(record.get("label") or "").strip().casefold()
            if not source_id or not label:
                continue
            previous = prompt_labels.setdefault(source_id, label)
            if previous != label:
                raise ValueError(f"Prompt file gives conflicting labels for source_id {source_id!r}.")

    result = manifest.copy()
    prompt_series = result["source_id"].astype(str).map(prompt_labels)
    has_prompt_label = result["split"].eq("train") & prompt_series.notna()
    result.loc[has_prompt_label, "source_label"] = prompt_series.loc[has_prompt_label]
    return _assign_project_labels(result)



def _filter_available_modalities(rows: pd.DataFrame, availability_columns: Sequence[str]) -> pd.DataFrame:
    eligible = rows.copy()
    for column in availability_columns:
        if column not in eligible:
            raise ValueError(f"Manifest is missing modality flag {column!r}.")
        eligible = eligible.loc[eligible[column]]
    return eligible


def load_manus_manifest(
    dataset_root: Path,
    config: str = "hagrid_v2",
) -> pd.DataFrame:
    """Đọc manifest đã tải và chuẩn hóa các cột dùng trong thí nghiệm."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    manifest_path = dataset_root / "metadata" / f"{config}_manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MANUS-HaGRID manifest not found: {manifest_path}.")
    manifest = _normalize_manifest(pd.read_parquet(manifest_path), config)
    prompt_path = dataset_root / f"HaGRID_{config.removeprefix('hagrid_').upper()}_train_prompt.jsonl"
    manifest = _join_prompt_labels(manifest, prompt_path)
    return _assign_project_labels(manifest)


def load_manus_manifest_from_hub(
    repo_id: str = "QFun/MANUS-HaGRID",
    config: str = "hagrid_v2",
    revision: str = "main",
) -> pd.DataFrame:
    """Tải manifest và prompt train nhỏ từ Hugging Face rồi nối nhãn theo source_id."""
    from huggingface_hub import hf_hub_download

    manifest_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"metadata/{config}_manifest.parquet",
        repo_type="dataset",
        revision=revision,
    )
    manifest = _normalize_manifest(pd.read_parquet(manifest_path), config)
    blank_train = manifest["split"].eq("train") & manifest["source_label"].fillna("").str.strip().eq("")
    version = config.removeprefix("hagrid_").upper()
    prompt_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"HaGRID_{version}_train_prompt.jsonl",
        repo_type="dataset",
        revision=revision,
    )
    manifest = _join_prompt_labels(manifest, Path(prompt_path))
    return _assign_project_labels(manifest)


def split_train_validation(
    manifest: pd.DataFrame,
    validation_fraction: float = 0.2,
    seed: int = 42,
    group_column: str = "source_id",
    availability_columns: Sequence[str] = ("has_hand_rgb",),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chia riêng train thành train/validation và giữ source group ở cùng một tập."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if "split" not in manifest or "project_label" not in manifest:
        raise ValueError("manifest must contain split and project_label columns.")
    if group_column not in manifest:
        group_column = "sample_id"
    if group_column not in manifest:
        raise ValueError(f"manifest is missing grouping column {group_column!r}.")

    train_rows = manifest.loc[
        (manifest["split"] == "train")
        & manifest["project_label"].notna()
    ].copy()
    train_rows = _filter_available_modalities(train_rows, availability_columns)
    if train_rows.empty:
        raise ValueError("No labeled train rows with hand_rgb crops are available.")

    group_class_counts = train_rows.groupby(group_column, dropna=False)["project_label"].nunique()
    if (group_class_counts > 1).any():
        raise ValueError("A source group maps to multiple project labels; inspect the manifest before splitting.")

    rng = np.random.default_rng(seed)
    validation_groups = set()
    for _, class_rows in train_rows.groupby("project_label", sort=True):
        groups = class_rows[group_column].dropna().astype(str).unique()
        if len(groups) < 2:
            raise ValueError("Each class needs at least two distinct source groups for validation splitting.")
        shuffled = rng.permutation(groups)
        validation_count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
        validation_groups.update(shuffled[:validation_count])

    validation_mask = train_rows[group_column].astype(str).isin(validation_groups)
    train_part = train_rows.loc[~validation_mask].copy().reset_index(drop=True)
    validation_part = train_rows.loc[validation_mask].copy().reset_index(drop=True)
    return train_part, validation_part


def balanced_subset(
    rows: pd.DataFrame,
    samples_per_class: int,
    seed: int = 42,
    label_column: str = "project_label",
    availability_columns: Sequence[str] = ("has_hand_rgb",),
) -> pd.DataFrame:
    """Lấy tập con cân bằng theo lớp với thứ tự chọn có thể tái tạo."""
    if not isinstance(samples_per_class, int) or samples_per_class < 1:
        raise ValueError("samples_per_class must be a positive integer.")
    if label_column not in rows:
        raise ValueError(f"rows must contain {label_column!r}.")
    eligible = rows.loc[rows[label_column].notna()].copy()
    eligible = _filter_available_modalities(eligible, availability_columns)
    counts = eligible[label_column].value_counts()
    if counts.empty:
        raise ValueError("No labeled rows are available for balanced sampling.")
    selected_count = min(samples_per_class, int(counts.min()))
    selected = [
        group.sample(n=selected_count, random_state=seed)
        for _, group in eligible.groupby(label_column, sort=True)
    ]
    return pd.concat(selected, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def select_shards_for_quota(
    rows: pd.DataFrame,
    samples_per_class: int,
    seed: int = 42,
    label_column: str = "project_label",
    availability_columns: Sequence[str] = ("has_hand_rgb",),
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """Chọn tham lam các shard cần thiết để đạt quota cho từng lớp."""
    if not isinstance(samples_per_class, int) or samples_per_class < 1:
        raise ValueError("samples_per_class must be a positive integer.")
    if not {"shard", label_column}.issubset(rows.columns):
        raise ValueError(f"rows must contain 'shard' and {label_column!r}.")
    eligible = rows.loc[rows[label_column].notna()].copy()
    eligible = _filter_available_modalities(eligible, availability_columns)
    class_labels = sorted(eligible[label_column].unique().tolist())
    if not class_labels:
        raise ValueError("No labeled rows are available for shard selection.")

    counts = pd.crosstab(eligible["shard"], eligible[label_column]).reindex(columns=class_labels, fill_value=0)
    if (counts.sum(axis=0) < samples_per_class).any():
        insufficient = counts.sum(axis=0).loc[counts.sum(axis=0) < samples_per_class].to_dict()
        raise ValueError(f"Not enough samples for requested per-class quota: {insufficient}")

    rng = np.random.default_rng(seed)
    shard_order = rng.permutation(counts.index.to_numpy())
    remaining = counts.reindex(shard_order).copy()
    collected = pd.Series(0, index=class_labels, dtype=np.int64)
    selected_shards = []
    while (collected < samples_per_class).any():
        deficits = (samples_per_class - collected).clip(lower=0)
        scores = remaining.apply(lambda row: np.minimum(row.to_numpy(), deficits.to_numpy()).sum(), axis=1)
        if scores.empty or scores.max() <= 0:
            raise ValueError("Could not find enough shards to meet the requested class quota.")
        selected = scores.idxmax()
        selected_shards.append(selected)
        collected += remaining.loc[selected]
        remaining = remaining.drop(index=selected)

    selected_rows = eligible.loc[eligible["shard"].isin(selected_shards)].copy().reset_index(drop=True)
    return selected_rows, selected_shards


def _resolve_shard_path(value: str, dataset_root: Path) -> Path:
    relative = Path(str(value))
    candidates = [relative] if relative.is_absolute() else [dataset_root / relative]
    if not relative.is_absolute():
        candidates.append(dataset_root / "data" / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"WebDataset shard not found for manifest value {value!r} under {dataset_root}.")


def load_hand_rgb_samples(rows: pd.DataFrame, dataset_root: Path) -> Dict[str, np.ndarray]:
    """Đọc các ảnh ``hand_rgb.png`` đã chọn từ shard WebDataset cục bộ.

    Trả về ánh xạ từ ``sample_id`` sang mảng ảnh BGR uint8. Chỉ mở các shard
    được tham chiếu trong ``rows``; hàm này không tải dữ liệu.
    """
    required = {"sample_id", "shard", "has_hand_rgb"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"rows is missing required columns: {sorted(missing)}")
    if rows["sample_id"].duplicated().any():
        raise ValueError("rows must contain unique sample IDs.")
    if not rows["has_hand_rgb"].all():
        raise ValueError("Selected rows include samples without hand_rgb crops.")

    dataset_root = Path(dataset_root).expanduser().resolve()
    requested_by_shard = {
        str(shard): set(group["sample_id"].astype(str))
        for shard, group in rows.groupby("shard", sort=True)
    }
    images: Dict[str, np.ndarray] = {}
    for shard_value, requested_ids in requested_by_shard.items():
        shard_path = _resolve_shard_path(shard_value, dataset_root)
        with tarfile.open(shard_path, mode="r:*") as archive:
            for member in archive:
                if not member.isfile() or "." not in Path(member.name).name:
                    continue
                sample_id, suffix = Path(member.name).name.split(".", 1)
                if sample_id not in requested_ids or suffix != "hand_rgb.png":
                    continue
                file_object = archive.extractfile(member)
                if file_object is None:
                    continue
                encoded = np.frombuffer(file_object.read(), dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    raise ValueError(f"Could not decode hand_rgb crop for {sample_id}.")
                images[sample_id] = image
                requested_ids.remove(sample_id)
                if not requested_ids:
                    break
        if requested_ids:
            raise FileNotFoundError(
                f"Missing hand_rgb crops for {len(requested_ids)} samples in {shard_path}: "
                f"{sorted(requested_ids)[:5]}"
            )
    return images


def load_hand_rgb_samples_from_hub(
    rows: pd.DataFrame,
    repo_id: str = "QFun/MANUS-HaGRID",
    revision: str = "main",
    timeout: Tuple[int, int] = (30, 300),
    grayscale: bool = False,
) -> Dict[str, np.ndarray]:
    """Stream các shard được chọn từ Hugging Face và giải mã crop bàn tay.

    Tar không được lưu cục bộ. Chỉ các shard được tham chiếu trong ``rows`` mới
    được đọc; dữ liệu vẫn tuân theo giấy phép upstream của HaGRID.
    """
    import requests
    from huggingface_hub import hf_hub_url

    required = {"sample_id", "shard", "has_hand_rgb"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"rows is missing required columns: {sorted(missing)}")
    if rows["sample_id"].duplicated().any():
        raise ValueError("rows must contain unique sample IDs.")
    if not rows["has_hand_rgb"].all():
        raise ValueError("Selected rows include samples without hand_rgb crops.")

    requested_by_shard = {
        str(shard): set(group["sample_id"].astype(str))
        for shard, group in rows.groupby("shard", sort=True)
    }
    images: Dict[str, np.ndarray] = {}
    for shard_path, requested_ids in requested_by_shard.items():
        url = hf_hub_url(
            repo_id=repo_id,
            filename=shard_path.replace("\\", "/").lstrip("/"),
            repo_type="dataset",
            revision=revision,
        )
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            with tarfile.open(fileobj=response.raw, mode="r|*") as archive:
                for member in archive:
                    if not member.isfile() or "." not in Path(member.name).name:
                        continue
                    sample_id, suffix = Path(member.name).name.split(".", 1)
                    if sample_id not in requested_ids or suffix != "hand_rgb.png":
                        continue
                    file_object = archive.extractfile(member)
                    if file_object is None:
                        continue
                    encoded = np.frombuffer(file_object.read(), dtype=np.uint8)
                    read_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
                    image = cv2.imdecode(encoded, read_mode)
                    if image is None or image.size == 0:
                        raise ValueError(f"Could not decode hand_rgb crop for {sample_id}.")
                    images[sample_id] = image
                    requested_ids.remove(sample_id)
                    if not requested_ids:
                        break
        if requested_ids:
            raise FileNotFoundError(
                f"Missing hand_rgb crops for {len(requested_ids)} samples in Hub shard {shard_path}: "
                f"{sorted(requested_ids)[:5]}"
            )
    return images


