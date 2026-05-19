from fastapi import FastAPI

from app.schemas import PredictRequest

from app.predictors.fear_predictor import predict_fear
from app.predictors.difficulty_predictor import predict_difficulty


app = FastAPI(
    title="Smart Filter API"
)


@app.get("/")
def healthcheck():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(request: PredictRequest):

    text = request.text

    fear_result = predict_fear(text)

    difficulty_result = predict_difficulty(text)

    return {
        "text": text,
        "fear": fear_result,
        "difficulty": difficulty_result
    }