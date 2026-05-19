import joblib

fear_model = joblib.load(
    "app/models/fear_model.joblib"
)

difficulty_model = joblib.load(
    "app/models/difficulty_model.joblib"
)