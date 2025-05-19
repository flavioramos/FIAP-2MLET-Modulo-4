import pytest
import os
import sys
import multiprocessing

# Set multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as flask_app

@pytest.fixture
def app():
    """Create and configure a Flask app for testing."""
    os.environ["LOCAL_RUN"] = "true"
    flask_app.config.update({
        "TESTING": True,
        "JWT_SECRET_KEY": "test-secret-key"
    })
    yield flask_app
    # Clean up environment variable after tests
    os.environ.pop("LOCAL_RUN", None)

@pytest.fixture
def client(app):
    """Create a test client for the Flask app."""
    return app.test_client()

@pytest.fixture
def auth_headers(client):
    """Get authentication headers for protected endpoints."""
    response = client.post("/login", json={
        "username": "user",
        "password": "password"
    })
    assert response.status_code == 200
    token = response.json["access_token"]
    return {"Authorization": f"Bearer {token}"} 