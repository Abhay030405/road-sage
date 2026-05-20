"""Phase 1 integration tests for the FastAPI application."""

from __future__ import annotations

import httpx  # imported per spec; used internally by TestClient
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Provide a TestClient that runs the full app lifespan (startup/shutdown)."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Meta / infrastructure endpoints
# ---------------------------------------------------------------------------


def test_root_endpoint(client):
    """GET / must return HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200


def test_root_contains_message(client):
    """GET / response body must include a 'message' field pointing to /docs."""
    response = client.get("/")
    body = response.json()
    assert "message" in body
    assert "RoadSage" in body["message"]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_health_endpoint(client):
    """GET /api/v1/health must return HTTP 200 with a 'status' field."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert body["status"] == "healthy"


def test_health_contains_version(client):
    """Health response must report the API version."""
    response = client.get("/api/v1/health")
    body = response.json()
    assert "version" in body
    assert body["version"] == "1.0.0"


def test_health_contains_uptime(client):
    """Health response must include a non-negative uptime_seconds value."""
    response = client.get("/api/v1/health")
    body = response.json()
    assert "uptime_seconds" in body
    assert isinstance(body["uptime_seconds"], (int, float))
    assert body["uptime_seconds"] >= 0


def test_health_models_not_loaded_phase1(client):
    """Phase 1: models_loaded must be False (inference not yet implemented)."""
    response = client.get("/api/v1/health")
    body = response.json()
    assert "models_loaded" in body
    assert body["models_loaded"] is False


def test_health_config_valid(client):
    """Config must load and validate successfully in the test environment."""
    response = client.get("/api/v1/health")
    body = response.json()
    assert "config_valid" in body
    assert body["config_valid"] is True


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------


def test_metrics_endpoint(client):
    """GET /metrics must return HTTP 200 (Prometheus scrape endpoint is up)."""
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_content_type(client):
    """Prometheus metrics endpoint must return a text/plain content type."""
    response = client.get("/metrics")
    assert "text/plain" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Swagger UI
# ---------------------------------------------------------------------------


def test_docs_available(client):
    """GET /docs must return HTTP 200 — Swagger UI must be accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
