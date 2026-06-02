import joblib
import numpy as np

MODEL_PATH = "app/models/age_model.joblib"

age_model = joblib.load(MODEL_PATH)


def predict_age(text: str) -> dict:
    text = str(text).strip().lower()

    pred_age = age_model.predict([text])[0]

    result = {
        "age": pred_age
    }

    if hasattr(age_model, "predict_proba"):
        probas = age_model.predict_proba([text])[0]
        classes = age_model.named_steps["clf"].classes_

        confidence = float(np.max(probas))

        result["confidence"] = round(confidence, 4)
        result["probabilities"] = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(classes, probas)
        }

    return result