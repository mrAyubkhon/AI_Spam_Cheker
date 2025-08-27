from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI(title="SMS Spam Detector API")

class Query(BaseModel):
    text: str

model = joblib.load(Path(__file__).parent / "model" / "sms_spam_pipeline.joblib")

@app.get("/")
def root():
    return {"status": "ok", "message": "SMS Spam Detector API"}

@app.post("/predict")
def predict(q: Query):
    pred = model.predict([q.text])[0]
    proba = model.predict_proba([q.text])[0].max().item()
    return {"label": pred, "confidence": round(float(proba), 4)}