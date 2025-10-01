# from contextlib import asynccontextmanager
# from pydantic import BaseModel
# from typing import Annotated, Union
# from fastapi import FastAPI, HTTPException
# import joblib
# import os
# import pickle
# import numpy as np
# import asyncio
# import time
# from config import LOGISTIC_MODEL, RF_MODEL

# labels = ["setosa", "versicolor", "virginica"]
# ml_models = {}

# # data model for imput


# class InputData(BaseModel):
#     data: list[list[float]]


# def load_model(path: str):
#     model = None
#     with open(path, "rb") as f:
#         model = joblib.load(f)
#     return model

# # Create a lifespan


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     app.state.logistic_reg = load_model(LOGISTIC_MODEL)
#     app.state.randomforest = load_model(RF_MODEL)
#     ml_models["logistic_model"] = load_model(LOGISTIC_MODEL)
#     ml_models["rf_model"] = load_model(RF_MODEL)
#     yield
# # Create a FastAPI instance
# app = FastAPI(lifespan=lifespan)

# # Health check endpoint


# @app.get("/health")
# async def health_check():
#     return {"status": "Healthy"}


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}


# @app.post("/predict/LogisticReg")
# async def prediction_lr(feature: InputData):
#     model = app.state.logistic_reg
#     pred = model.predict(np.array(feature.data))

#     return {"Prediction label": pred.tolist(), "name:": labels[pred[0]]}


# @app.post("/predict/LogistiRandomClassifer")
# async def prediction_rf(feature: InputData):

#     model = app.state.randomforest
#     print("TEST: ", np.array(feature.data))
#     pred = model.predict(np.array(feature.data))
#     return {"Prediction label": pred.tolist(), "name:": labels[pred[0]]}

# # Task 2.3: Create a GET endpoint to list available models


# @app.get("/models")
# async def list_models():
#     return {"models": list(ml_models.keys())}

# # Part 3: Building a Simple Prediction API


# class IrisData(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float

# # Add a POST endpoint for model predictions:


# @app.post("/predict/{model_name}")
# async def predict(model_name: str, iris: IrisData):
#     input_data = [[iris.sepal_length, iris.sepal_width,
#                    iris.petal_length, iris.petal_width]]

#     if model_name not in ml_models.keys():
#         raise HTTPException(status_code=404, detail="Model not found")

#     ml_model = ml_models[model_name]
#     prediction = ml_model.predict(input_data)

#     return {"model": model_name, "prediction": int(prediction[0])}

# # Test the API using curl:
# ''' curl - X 'POST' \
#     'http://127.0.0.1:8000/predict/logistic_model' \
#     - H 'accept: application/json' \
#     - H 'Content-Type: application/json' \
#     - d '{
#         "sepal_length": 0.5,
#         "sepal_width": 2.4,
#         "petal_length": 3.2,
#         "petal_width": 4.0
#     }'''

# # Task 3.2: Adding Asynchronous Predictions

# # Simulate the long-running tasks:


# @app.post("/predict/{model_name}")
# async def predict(model_name: str, iris: IrisData):
#     await asyncio.sleep(5)
#     input_data = [[iris.sepal_length, iris.sepal_width,
#                    iris.petal_length, iris.petal_width]]

#     if model_name not in ml_models.keys():
#         raise HTTPException(status_code=404, detail="Model not found")

#     ml_model = ml_models[model_name]
#     prediction = ml_model.predict(input_data)

#     return {"model": model_name, "prediction": int(prediction[0])}

# # Test asynchronous behavior with multiple curl requests:
# ```bash
# curl - X 'POST' \
#     'http://127.0.0.1:8000/predict/logistic_model' \
#     - H 'accept: application/json' \
#      - H 'Content-Type: application/json' \
#     - d '{
#         "sepal_length": 0.5,
#         "sepal_width": 0.2,
#         "petal_length": 0.1,
#         "petal_width": 0.9
#     }' &
# curl - X 'POST' \
#     'http://localhost:8000/predict/rf_model' \
#     - H 'accept: application/json' \
#      - H 'Content-Type: application/json' \
#     - d '{
#         "sepal_length": 0.1,
#         "sepal_width": 0.1,
#         "petal_length": 0.1,
#         "petal_width": 0.1
#     }'
# ```python

# # class Item(BaseModel):
# #     name: str
# #     price: float
# #     is_offer: Union[bool, None] = None


# # @app.get("/items/{item_id}")
# # def read_item(item_id: int, q: Union[str, None] = None):
# #     return {"item_id": item_id, "q": q}


# # @app.put("/items/{item_id}")
# # def update_item(item_id: int, item: Item):
# #     return {"item_name": item.name, "item_id": item_id}


from pydantic import BaseModel, Field
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Path
from config import LOGISTIC_MODEL, RF_MODEL
from auth import require_api_key
from typing import Annotated
from contextlib import asynccontextmanager
import time
import pickle
import asyncio

app = FastAPI()


class IrisData(BaseModel):
    sepal_length: float = Field(
        default=1.1, gt=0, lt=10, description="Sepal length is in range (0,10)"
    )
    sepal_width: float = Field(default=3.1, gt=0, lt=10)
    petal_length: float = Field(default=2.1, gt=0, lt=10)
    petal_width: float = Field(default=4.1, gt=0, lt=10)


ml_models = {}  # Global dictionary to hold the models.


def load_model(path: str):
    if not path:
        return None
    model = None
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models when the app starts
    ml_models["logistic_model"] = load_model(LOGISTIC_MODEL)
    ml_models["rf_model"] = load_model(RF_MODEL)

    yield

    # This code will be executed after the application finishes handling requests, right before the shutdown
    # Clean up the ML models and release the resources
    ml_models.clear()


# Create a FastAPI instance
app = FastAPI(lifespan=lifespan)


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models")
async def list_models():
    print(LOGISTIC_MODEL)
    print(RF_MODEL)
    return {"available_models": list(ml_models.keys())}


@app.post("/predict/{model_name}")
async def predict(
    model_name: Annotated[str, Path(pattern=r"^(logistic_model|rf_model)$")],
    iris: IrisData,
    background_tasks: BackgroundTasks,
):
    # await asyncio.sleep(5) # Mimic heavy workload.

    input_data = [
        [iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]
    ]

    if model_name not in ml_models.keys():
        raise HTTPException(status_code=404, detail="Model not found")

    ml_model = ml_models[model_name]
    prediction = ml_model.predict(input_data)

    background_tasks.add_task(
        log_prediction,
        {"model": model_name, "features": iris.model_dump(), "prediction": prediction},
    )

    return {"model": model_name, "prediction": int(prediction[0])}


def log_prediction(data: dict):
    time.sleep(5)  # mimic heavy work.
    print(f"Logging prediction: {data}")


@app.post("/predict_secure/{model_name}")
async def predict_secure(
    model_name: Annotated[str, Path(pattern=r"^(logistic_model|rf_model)$")],
    iris: IrisData,
    background_tasks: BackgroundTasks,
    _: str = Depends(require_api_key),
):
    # await asyncio.sleep(5) # Mimic heavy workload.

    input_data = [
        [iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]
    ]

    if model_name not in ml_models.keys():
        raise HTTPException(status_code=404, detail="Model not found")

    ml_model = ml_models[model_name]
    prediction = ml_model.predict(input_data)

    background_tasks.add_task(
        log_prediction,
        {"model": model_name, "features": iris.dict(), "prediction": prediction},
    )

    return {"model": model_name, "prediction": int(prediction[0])}
