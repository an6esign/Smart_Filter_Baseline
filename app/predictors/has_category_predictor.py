from app.services.category_rules import extract_categories


def predict_has_category(text: str):

    categories = extract_categories(text)

    return {
        "prediction": 1 if categories else 0,
        "confidence": 1.0
    }
