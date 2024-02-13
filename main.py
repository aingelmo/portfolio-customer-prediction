from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pycaret.classification import load_model, predict_model
from pydantic import BaseModel


class Data(BaseModel):
    Year_Birth: int
    Education: Literal["Graduation", "PhD", "Master", "2n Cycle", "Basic"]
    Income: float


model_location = Path(__file__).parent / "models" / "gbr_mnt_wine"

app = FastAPI()

model = load_model(str(model_location))


@app.post("/predict_mnt_wine")
def predict(input_dict: Data):
    input_df = pd.DataFrame([input_dict.model_dump()])
    predictions_df = predict_model(estimator=model, data=input_df)
    prediction = predictions_df["prediction_label"].iloc[0]
    return int(prediction)
