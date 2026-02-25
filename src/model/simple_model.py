from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class LinearModel:
    weights: List[float]
    bias: float

    def predict(self, inputs: Sequence[Sequence[float]]) -> List[float]:
        outputs: List[float] = []
        for row in inputs:
            if len(row) != len(self.weights):
                raise ValueError("Input length does not match weights.")
            total = self.bias
            for weight, value in zip(self.weights, row):
                total += weight * value
            outputs.append(total)
        return outputs
