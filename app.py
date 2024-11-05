from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import uvicorn


app = FastAPI()

# Define input features[Important features only!]
class InputData(BaseModel):
    department: int
    review: float
    projects: int
    tenure: float
    satisfaction: float
    avg_hrs_month: float
    
# Load the model from the file
with open('models/RFC.pkl', 'rb') as file:
    model = pickle.load(file)

@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}

@app.post("/predict")
async def predict(input_data: InputData):
        df = pd.DataFrame(data = [input_data.model_dump().values()], 
                          columns = input_data.model_dump().keys())
        predictions = model.predict(df)
        return {"predicted_class": int(predictions[0])}