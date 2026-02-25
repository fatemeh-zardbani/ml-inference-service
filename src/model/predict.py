from __future__ import annotations

from typing import List, Mapping, Sequence, Union

from .load import ModelType, PreprocessConfig

InputRecord = Mapping[str, Union[int, float]]
InputData = Union[InputRecord, Sequence[InputRecord]]


def _normalize(values: Sequence[float], config: PreprocessConfig) -> List[float]:
    normalized: List[float] = []
    for value, mean, std in zip(values, config.mean, config.std):
        if std == 0:
            normalized.append(value - mean)
        else:
            normalized.append((value - mean) / std)
    return normalized


def _prepare_inputs(
    input_data: InputData, config: PreprocessConfig
) -> List[List[float]]:
    if isinstance(input_data, Mapping):
        records = [input_data]
    else:
        records = list(input_data)

    prepared: List[List[float]] = []
    for record in records:
        ordered = [float(record[feature]) for feature in config.feature_order]
        prepared.append(_normalize(ordered, config))

    return prepared


def predict(model: ModelType, input_data: InputData) -> List[float]:
    """Run deterministic inference with explicit preprocessing."""
    prepared = _prepare_inputs(input_data, model.preprocess)
    outputs = model.model.predict(prepared)
    return list(outputs)
