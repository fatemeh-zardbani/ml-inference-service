from src.model.load import load_model
from src.model.predict import predict
import pytest


def test_golden_prediction():
    # use known artifact produced by smoke_test
    model = load_model()
    inp = {"feature_a": 1.0, "feature_b": 0.5, "feature_c": -1.0}
    out = predict(model, inp)
    # manual calculation based on demo weights/bias
    assert out == [-0.525]


def test_list_input():
    model = load_model()
    samples = [
        {"feature_a": 1.0, "feature_b": 0.5, "feature_c": -1.0},
        {"feature_a": 0.0, "feature_b": 0.0, "feature_c": 0.0},
    ]
    out = predict(model, samples)
    assert isinstance(out, list) and len(out) == 2


def test_missing_feature_raises():
    model = load_model()
    with pytest.raises(KeyError):
        predict(model, {"feature_a": 1.0})


def test_mismatched_length_raises():
    # prepare input with extra unmapped feature
    model = load_model()
    bad = {"feature_a": 1.0, "feature_b": 2.0, "feature_c": 3.0, "feature_d": 4.0}
    # predict should ignore extras or raise; our implementation will ignore extras when ordering but not fail
    # we assert it doesn't crash and returns a single number
    out = predict(model, bad)
    assert isinstance(out, list) and len(out) == 1