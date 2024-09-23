from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from typing import Dict


model = joblib.load("model.pkl")


app = FastAPI()


class ModelInput(BaseModel):
    features: Dict[int, float]  


@app.post("/predict")
def predict(input_data: ModelInput):
    input_list = [list(input_data.features.values())]
    prediction = model.predict(input_list)

    
    label = "Fraud" if prediction[0] == 1.0 else "non-fraud"

    return {"prediction": label}



@app.get("/")
def read_root():
    return {"message": "API is running!"}
