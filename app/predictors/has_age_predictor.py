import numpy as np

from app.model_loader import has_age_model



def predict_has_age(text: str):

    pred = int(
        has_age_model.predict([text])[0]
    )

    probs = has_age_model.predict_proba([text])[0]

    confidence = float(np.max(probs))

    return {
        "prediction": pred,
        "confidence": round(confidence, 4)
    }