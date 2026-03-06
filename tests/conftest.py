import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# make project root, src, and app importable
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "app"))

from scripts.smoke_test import _ensure_demo_artifacts
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def setup_artifacts():
    # ensure model and preprocess files exist for tests
    _ensure_demo_artifacts()


@pytest.fixture

def client():
    return TestClient(app)