from fastapi import FastAPI

from app.schemas import PredictRequest

from app.predictors.fear_predictor import (
    predict_fear
)

from app.predictors.fear_level_predictor import (
    predict_fear_level
)

from app.predictors.difficulty_predictor import (
    predict_difficulty
)

from app.predictors.difficulty_level_predictor import (
    predict_difficulty_level
)

from app.predictors.has_format_predictor import (
    predict_has_format
)

from app.predictors.type_of_format_predictor import (
    predict_format
)

from app.predictors.category_predictor import (
    predict_category
)

from app.predictors.has_age_predictor import (
    predict_has_age
)

from app.predictors.age_predictor import (
    predict_age
)

from app.predictors.rubert_fear_predictor import predict_rubert_fear


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

    # =========================
    # FEAR
    # =========================

    fear_result = predict_fear(text)

    if fear_result["prediction"] == 1:

        fear_level_result = (
            predict_fear_level(text)
        )

    else:

        fear_level_result = {
            "prediction": "unknown",
            "confidence": 0.0
        }

    # =========================
    # DIFFICULTY
    # =========================

    difficulty_result = predict_difficulty(text)

    if difficulty_result["prediction"] == 1:

        difficulty_level_result = (
            predict_difficulty_level(text)
        )

    else:

        difficulty_level_result = {
            "prediction": "unknown",
            "confidence": 0.0
        }

    # =========================
    # FORMAT
    # =========================

    has_format_result = predict_has_format(
        text
    )

    if has_format_result["prediction"] == 1:

        format_result = predict_format(
            text
        )

    else:

        format_result = {
            "prediction": "unknown",
            "confidence": 0.0,
            "probabilities": {}
        }

    # =========================
    # CATEGORY
    # =========================

    category_result = predict_category(
        text
    )
    
# =========================
# AGE
# =========================

    has_age_result = predict_has_age(
        text
    )

    if has_age_result["prediction"] == 1:

        age_result = predict_age(
            text
        )

    else:

        age_result = {
            "age": "unknown",
            "confidence": 0.0
        }

    # =========================
    # RESPONSE
    # =========================

    return {

        "text": text,

        "fear": {

            "has_fear": fear_result,

            "fear_level": (
                fear_level_result
            )
        },

        "difficulty": {

            "has_difficulty": (
                difficulty_result
            ),

            "difficulty_level": (
                difficulty_level_result
            )
        },

        "format": {

            "has_format": (
                has_format_result
            ),

            "format_type": (
                format_result
            )
        },
        
        "age": {

            "has_age": (
                has_age_result
            ),

            "age": (
                age_result
            )
        },

        "category": category_result
    }
    
@app.post("/rubert_fear_predict")
def rubert_fear_predict(request: PredictRequest):

    result = predict_rubert_fear(
        request.text
    )

    return {
        "text": request.text,
        "prediction": result
    }