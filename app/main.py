from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Union
import sys
import time
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from model.load import load_model
from model.predict import predict

app = FastAPI()

# -- request models ---------------------------------------------------------

class FeatureRecord(BaseModel):
    feature_a: float = Field(..., description="first feature")
    feature_b: float = Field(..., description="second feature")
    feature_c: float = Field(..., description="third feature")

    @validator("feature_a", "feature_b", "feature_c")
    def within_reasonable_range(cls, v: float) -> float:
        # example range check; adjust as appropriate for your domain
        if not (-1e6 <= v <= 1e6):
            raise ValueError("feature value out of range")
        return v

class PredictRequest(BaseModel):
    data: Union[FeatureRecord, List[FeatureRecord]]

# -- error handlers ---------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # let FastAPI default behaviour handle status and detail, but unify format
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "internal server error"},
    )

# -- health -----------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy"}

# -- prediction -------------------------------------------------------------

@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # convert Pydantic models into plain dictionaries; predict expects mappings
    if isinstance(request.data, list):
        input_data = [record.dict() for record in request.data]
    else:
        input_data = request.data.dict()

    start = time.perf_counter()
    try:
        predictions = predict(model, input_data)
    except Exception as e:
        # propagate a standardized error message for callers
        raise HTTPException(status_code=500, detail=f"model inference error: {e}")
    latency_ms = (time.perf_counter() - start) * 1000

    # simulate a timeout condition for demo purposes
    if latency_ms > 5000:
        raise HTTPException(status_code=504, detail="inference timeout")

    response = {
        "predictions": predictions,
        "model_version": getattr(model, "version", "unknown"),
        "latency_ms": round(latency_ms, 2),
    }
    return response