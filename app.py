from fastapi import FastAPI, Request
from pydantic import BaseModel
from model import predict

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def get_prediction(input_data: InputData):
    result = predict(input_data.features)
    return {"prediction": result}
