from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np


app = FastAPI(
    title="Relationship Pattern Classifier",
    description="Detects communication patterns like lovebombing, gaslighting, etc.",
    version="1.0.0"
)

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


class MessageRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Input message")


class PredictionResponse(BaseModel):
    input_text: str
    predicted_label: str
    confidence: float


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty")

        text_vectorized = vectorizer.transform([request.text])

        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = float(np.max(probabilities))

        return PredictionResponse(
            input_text=request.text,
            predicted_label=prediction,
            confidence=round(confidence, 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))