from pydantic import BaseModel, Field
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Path
from contextlib import asynccontextmanager
from typing import Annotated
import pickle
import asyncio
import time
from config import LOGISTIC_MODEL, RF_MODEL
from auth import require_api_key

app = FastAPI()

# Global dictionary to hold the models
ml_models = {}


class IrisData(BaseModel):
    sepal_length: float = Field(gt=0, lt=10)
    sepal_width: float = Field(gt=0, lt=10)
    petal_length: float = Field(gt=0, lt=10)
    petal_width: float = Field(gt=0, lt=10)


def load_model(path: str):
    """Load a model from a .pkl file."""
    if not path:
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models when the app starts (production only)."""
    # Only load if the models are not already mocked (for tests)
    if not ml_models:
        ml_models["logistic_model"] = load_model(LOGISTIC_MODEL)
        ml_models["rf_model"] = load_model(RF_MODEL)
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


# Health and root endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models")
async def list_models():
    return {"available_models": list(ml_models.keys())}


# Prediction endpoint
@app.post("/predict/{model_name}")
async def predict(
    model_name: Annotated[str, Path(pattern=r"^(logistic_model|rf_model)$")],
    iris: IrisData,
    background_tasks: BackgroundTasks,
):
    input_data = [[iris.sepal_length, iris.sepal_width,
                   iris.petal_length, iris.petal_width]]

    if model_name not in ml_models:
        raise HTTPException(status_code=404, detail="Model not found")

    ml_model = ml_models[model_name]
    prediction = ml_model.predict(input_data)

    # Optional logging task
    background_tasks.add_task(
        log_prediction,
        {"model": model_name, "features": iris.model_dump(), "prediction": prediction}
    )

    return {"model": model_name, "prediction": int(prediction[0])}


def log_prediction(data: dict):
    """Simulate heavy logging work."""
    time.sleep(5)
    print(f"Logging prediction: {data}")


@app.post("/predict_secure/{model_name}")
async def predict_secure(
    model_name: Annotated[str, Path(pattern=r"^(logistic_model|rf_model)$")],
    iris: IrisData,
    background_tasks: BackgroundTasks,
    _: str = Depends(require_api_key),
):
    input_data = [[iris.sepal_length, iris.sepal_width,
                   iris.petal_length, iris.petal_width]]

    if model_name not in ml_models:
        raise HTTPException(status_code=404, detail="Model not found")

    ml_model = ml_models[model_name]
    prediction = ml_model.predict(input_data)

    background_tasks.add_task(
        log_prediction,
        {"model": model_name, "features": iris.dict(), "prediction": prediction}
    )

    return {"model": model_name, "prediction": int(prediction[0])}
