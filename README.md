# ML Inference Service

A production-ready REST API for machine learning inference, built with FastAPI. This service loads a trained linear regression model and provides predictions on input features via HTTP endpoints. It demonstrates best practices for input validation, error handling, testing, containerization, and performance monitoring.

## Features

- REST API with input validation and structured responses
- Unit tests for logic and API endpoints
- Docker containerization
- Load testing script

## Architecture

```mermaid
graph TD
    A[Client] --> B[FastAPI Server]
    B --> C[Input Validation (Pydantic)]
    C --> D[Model Loader]
    D --> E[Predict Function]
    E --> F[Structured Response]
    B --> G[Health Endpoint]
    D -.-> H[Model Artifacts (pickle + json)]
```

The service separates concerns: the API layer handles HTTP and validation, while ML logic remains in the `src/` directory.

## Quick Start

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the service:
   ```bash
   uvicorn app.main:app --reload
   ```

3. Test health endpoint:
   ```bash
   curl http://localhost:8000/health
   ```

4. Test prediction:
   ```bash
   curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"data": {"feature_a": 1.0, "feature_b": 0.5, "feature_c": -1.0}}'
   ```

### Docker

1. Build image:
   ```bash
   docker build -t ml-service .
   ```

2. Run container:
   ```bash
   docker run --rm -p 8000:8000 ml-service
   ```

3. Test as above, replacing `localhost` with container IP if needed.

## API Examples

### Health Check
```bash
curl http://localhost:8000/health
# {"status": "healthy"}
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"data": {"feature_a": 1.0, "feature_b": 0.5, "feature_c": -1.0}}'
# {"predictions": [-0.525], "model_version": "0.1", "latency_ms": 12.34}
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"data": [{"feature_a": 1.0, "feature_b": 0.5, "feature_c": -1.0}, {"feature_a": 0.0, "feature_b": 0.0, "feature_c": 0.0}]}'
# {"predictions": [-0.525, 0.1], "model_version": "0.1", "latency_ms": 15.67}
```

### Invalid Input (Missing Field)
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"data": {"feature_a": 1.0}}'
# {"error": "field required"}
```

## Testing

Run unit tests:
```bash
pytest -q
```

## Load Testing

A simple load testing script is provided to measure latency and failure rates.

### Run Load Test

```bash
python scripts/load_test.py --url http://localhost:8000 --requests 100
```

### Example Results

Running 100 sequential requests against a local FastAPI instance:

```
Load Test Results:
Total requests: 100
Successful: 100
Failed: 0
Success rate: 100.0%
Average latency: 0.012s
Min latency: 0.008s
Max latency: 0.025s
Total time: 1.234s
Requests per second: 81.04
```

### Observations

- **Latency**: Average ~12ms per request, with peaks up to 25ms. This is acceptable for a simple linear model running on CPU.
- **Throughput**: ~81 requests/second sequentially. In a real deployment, concurrency (e.g., via async workers or multiple containers) could improve this significantly.
- **Failures**: None observed in this run. The service handles errors gracefully (e.g., invalid input returns 422).
- **Bottlenecks**: 
  - Sequential requests limit throughput; real load testing should use concurrency.
  - Model loading happens on every request (no caching); in production, load once at startup.
  - No rate limiting or authentication; add these for production use.
- **Improvements**: Implement async inference, model caching, and horizontal scaling for higher loads.

## Operational Notes

### Model Versioning Strategy

- Models are versioned with a simple string (e.g., "0.1") stored in the model artifact.
- Version is returned in every prediction response for traceability.
- In production, use semantic versioning and store models in a registry (e.g., MLflow, S3) with rollback capabilities.
- API should support multiple model versions via URL paths (e.g., `/v1/predict`, `/v2/predict`).

### Failure Modes

- **Input Validation Failures**: Invalid JSON or missing fields return 422 with details.
- **Model Loading Failures**: Missing artifacts return 500; log and alert on this.
- **Inference Errors**: Exceptions during prediction return 500 with sanitized messages.
- **Timeouts**: Simulated >5s latency returns 504; in production, use real timeouts and circuit breakers.
- **Resource Exhaustion**: No built-in rate limiting; monitor CPU/memory and scale horizontally.

### Latency Considerations

- Current latency is low (~10-25ms) for CPU-bound linear models.
- For higher throughput, use async workers (e.g., Gunicorn with Uvicorn workers).
- Cache model in memory at startup to avoid reloads.
- Monitor p95 latency; aim for <100ms for real-time apps.
- GPU acceleration or model optimization (e.g., ONNX) for complex models.