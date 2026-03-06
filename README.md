# ml-inference-service

A simple ML inference service built with FastAPI, demonstrating production-ready practices.

## Features

- REST API with input validation and structured responses
- Unit tests for logic and API endpoints
- Docker containerization
- Load testing script

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