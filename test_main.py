from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from main import app, ml_models


@pytest.fixture(autouse=True)
def setup_ml_models():
    """Mock ML models before each test."""
    ml_models["logistic_model"] = MagicMock()
    ml_models["rf_model"] = MagicMock()

    # Set default predict return values
    ml_models["rf_model"].predict.return_value = [0]
    ml_models["logistic_model"].predict.return_value = [-1]

    yield

    # Clear models after test
    ml_models.clear()


def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello World"}


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_list_models():
    with TestClient(app) as client:
        response = client.get("/models")
        assert response.status_code == 200
        assert set(response.json()["available_models"]) == {
            "logistic_model", "rf_model"}


def test_predict_valid_model():
    with TestClient(app) as client:
        response = client.post(
            "/predict/rf_model",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )
        assert response.status_code == 200
        assert response.json() == {"model": "rf_model", "prediction": 0}


def test_predict_invalid_model():
    with TestClient(app) as client:
        response = client.post(
            "/predict/invalid_model",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )
        assert response.status_code == 422


def test_predict_mocked_logistic_model():
    with TestClient(app) as client:
        response = client.post(
            "/predict/logistic_model",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )
        assert response.status_code == 200
        assert response.json() == {"model": "logistic_model", "prediction": -1}
