from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "xgb_fatal_predictorv.pkl"

print("Loading model from:", MODEL_PATH)

model = joblib.load(MODEL_PATH)

app = FastAPI(title="AccidentIQ", version="1.0")

class AccidentInput(BaseModel):
    road_type: int
    speed_limit: int
    urban_or_rural_area: int
    junction_detail: int
    junction_control: int
    light_conditions: int
    weather_conditions: int
    road_surface_conditions: int
    carriageway_hazards: int
    number_of_vehicles: int
    number_of_casualties: int
    vehicle_type_mode: int
    any_skidding: int
    avg_vehicle_age: float
    time_sin: float
    time_cos: float
    day_sin: float
    day_cos: float

@app.get("/")
def home():
    return {"message": "AccidentIQ API is running"}

@app.post("/predict")
def predict(data: AccidentInput):

    # convert input to numpy array in correct feature order

    features = np.array([[
        data.road_type,
        data.speed_limit,
        data.urban_or_rural_area,
        data.junction_detail,
        data.junction_control,
        data.light_conditions,
        data.weather_conditions,
        data.road_surface_conditions,
        data.carriageway_hazards,
        data.number_of_vehicles,
        data.number_of_casualties,
        data.vehicle_type_mode,
        data.any_skidding,
        data.avg_vehicle_age,
        data.time_sin,
        data.time_cos,
        data.day_sin,
        data.day_cos
    ]])

    # get Fatal probability

    fatal_probability = model.predict_proba(features)[0][1]

    # determine risk level from probability

    if fatal_probability >= 0.4:
        risk_level = "High"
    elif fatal_probability >= 0.2:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "fatal_probability": round(float(fatal_probability), 3),
        "risk_level": risk_level,
        "model_version": "1.0"
    }