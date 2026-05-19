import numpy as np

from app.model_loader import difficulty_model



def predict_difficulty(text: str):

    pred = int(
        difficulty_model.predict([text])[0]
    )

    probs = difficulty_model.predict_proba([text])[0]

    confidence = float(np.max(probs))

    return {
        "prediction": pred,
        "confidence": round(confidence, 4)
    }