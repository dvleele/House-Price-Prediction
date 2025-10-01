from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Load model
model = joblib.load("../src/house_price_model.pkl")

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
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
