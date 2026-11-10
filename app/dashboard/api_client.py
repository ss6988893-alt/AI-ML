"""Thin Streamlit-side API client; prediction logic stays in FastAPI."""
import os
import httpx

BASE_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

def get(path: str, params: dict | None = None):
    response = httpx.get(f"{BASE_URL}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()

def post(path: str, payload: dict):
    response = httpx.post(f"{BASE_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()
