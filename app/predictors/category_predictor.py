from app.services.category_rules import extract_categories


def predict_category(text: str):

    predicted_categories = extract_categories(text)

    return {
        "prediction": predicted_categories,
        "confidence": 1.0 if predicted_categories else 0.0
    }
