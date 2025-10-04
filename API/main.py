from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import os
import uvicorn

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "src", "house_price_model.pkl")
model_path = os.path.abspath(model_path)

model = joblib.load(model_path)

app = FastAPI()

# Input schema
class HouseFeatures(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float
    FullBath: int

@app.post("/predict")
def predict_price(features: HouseFeatures):
    data = np.array([[features.OverallQual, features.GrLivArea,
                      features.GarageCars, features.TotalBsmtSF, features.FullBath]])
    prediction = model.predict(data)[0]
    return {"predicted_price": prediction}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
