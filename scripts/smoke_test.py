from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.model.load import load_model
from src.model.predict import predict
from src.model.simple_model import LinearModel

ARTIFACT_DIR = PROJECT_ROOT / "app" / "model_artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
PREPROCESS_PATH = ARTIFACT_DIR / "preprocess.json"


def _ensure_demo_artifacts() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and PREPROCESS_PATH.exists():
        return

    preprocess = {
        "feature_order": ["feature_a", "feature_b", "feature_c"],
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    }
    model = LinearModel(weights=[0.5, -0.25, 1.0], bias=0.1, version="0.1")

    with MODEL_PATH.open("wb") as handle:
        pickle.dump(model, handle)

    with PREPROCESS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(preprocess, handle, indent=2)


def main() -> None:
    _ensure_demo_artifacts()
    model = load_model()

    samples = [
        {"feature_a": 1.0, "feature_b": 0.5, "feature_c": -1.0},
        {"feature_a": 0.2, "feature_b": 1.3, "feature_c": 0.0},
        {"feature_a": -0.4, "feature_b": 0.1, "feature_c": 2.0},
    ]

    outputs = predict(model, samples)
    for index, output in enumerate(outputs, start=1):
        print(f"Prediction {index}: {output:.4f}")


if __name__ == "__main__":
    main()
