import pytest


def test_predict_valid(client):
    payload = {"data": {"feature_a": 1.0, "feature_b": 0.5, "feature_c": -1.0}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert "model_version" in body
    assert "latency_ms" in body
    assert body["predictions"] == [-0.525]


def test_predict_invalid_missing(client):
    payload = {"data": {"feature_a": 1.0}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_type(client):
    payload = {"data": {"feature_a": "nope", "feature_b": 0, "feature_c": 0}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_out_of_range(client):
    payload = {"data": {"feature_a": 1e7, "feature_b": 0, "feature_c": 0}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_batch(client):
    payload = {"data": [
        {"feature_a": 0.1, "feature_b": 0.2, "feature_c": 0.3},
        {"feature_a": 0.4, "feature_b": 0.5, "feature_c": 0.6},
    ]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert isinstance(r.json()["predictions"], list)
    assert len(r.json()["predictions"]) == 2
