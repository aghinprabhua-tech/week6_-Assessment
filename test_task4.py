import os
from dotenv import load_dotenv


# Load environment variables from .env file so the test uses the exact same key as task4.py
load_dotenv()


import pytest
from fastapi.testclient import TestClient
from Task_4 import app


client = TestClient(app)


# Dynamically fetch the API key from the environment (defaulting to the secret key if not found)
CURRENT_API_KEY = os.getenv("API_KEY", "secret_api_key_123")


VALID_HEADERS = {"X-API-Key": CURRENT_API_KEY}
INVALID_HEADERS = {"X-API-Key": "wrong_key"}


def test_health_check():
    """Test GET /health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_search_unauthorized():
    """Test POST /search without API key returns validation error."""
    response = client.post("/search", json={"query": "AI", "top_k": 5})
    assert response.status_code == 422


def test_search_invalid_api_key():
    """Test POST /search with invalid API key returns 401 Unauthorized."""
    response = client.post("/search", headers=INVALID_HEADERS, json={"query": "AI", "top_k": 5})
    assert response.status_code == 401


def test_search_valid_request():
    """Test POST /search with valid query and parameters returns results."""
    response = client.post(
        "/search",
        headers=VALID_HEADERS,
        json={"query": "machine learning", "top_k": 3, "min_score": 0.0}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_search_validation_top_k_out_of_bounds():
    """Test POST /search with top_k > 20 fails Pydantic validation (422)."""
    response = client.post(
        "/search",
        headers=VALID_HEADERS,
        json={"query": "test", "top_k": 25}
    )
    assert response.status_code == 422


def test_upload_document():
    """Test POST /documents successfully indexes a new document."""
    response = client.post(
        "/documents",
        headers=VALID_HEADERS,
        json={"source": "test_policy.pdf", "doc_type": "PDF", "text": "Sample policy content text."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"