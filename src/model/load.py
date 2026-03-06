from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "app" / "model_artifacts"


@dataclass(frozen=True)
class PreprocessConfig:
    feature_order: List[str]
    mean: List[float]
    std: List[float]


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    preprocess: PreprocessConfig


ModelType = LoadedModel


def _load_preprocess_config(path: Path) -> PreprocessConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    feature_order = list(raw.get("feature_order", []))
    mean = [float(value) for value in raw.get("mean", [])]
    std = [float(value) for value in raw.get("std", [])]

    if not feature_order:
        raise ValueError("Preprocess config missing feature_order.")
    if len(mean) != len(feature_order) or len(std) != len(feature_order):
        raise ValueError("Preprocess config sizes must match feature_order.")

    return PreprocessConfig(feature_order=feature_order, mean=mean, std=std)


def load_model(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> ModelType:
    """Load the serialized model and preprocessing config from the given directory."""
    model_path = artifact_dir / "model.pkl"
    preprocess_path = artifact_dir / "preprocess.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}.")
    if not preprocess_path.exists():
        raise FileNotFoundError(f"Preprocess config not found at {preprocess_path}.")

    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    preprocess = _load_preprocess_config(preprocess_path)

    return LoadedModel(model=model, preprocess=preprocess)
    if not PREPROCESS_PATH.exists():
        raise FileNotFoundError(
            f"Preprocess config not found at {PREPROCESS_PATH}."
        )

    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)

    preprocess = _load_preprocess_config(PREPROCESS_PATH)

    return LoadedModel(model=model, preprocess=preprocess)
