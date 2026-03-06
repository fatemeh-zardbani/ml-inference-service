from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Union
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from model.load import load_model
from model.predict import predict

app = FastAPI()

class PredictRequest(BaseModel):
    data: Union[Dict[str, float], List[Dict[str, float]]]

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    model = load_model()
    predictions = predict(model, request.data)
    return {"predictions": predictions}