import numpy as np

from app.model_loader import fear_model



def predict_fear(text: str):

    pred = int(
        fear_model.predict([text])[0]
    )

    probs = fear_model.predict_proba([text])[0]

    confidence = float(np.max(probs))

    return {
        "prediction": pred,
        "confidence": round(confidence, 4)
    }