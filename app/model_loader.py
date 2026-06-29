from pathlib import Path
import joblib

# =========================
# BASE DIR
# =========================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"

# =========================
# FEAR
# =========================

fear_model = joblib.load(
    MODELS_DIR / "fear_model.joblib"
)

fear_level_model = joblib.load(
    MODELS_DIR / "fear_level_model.joblib"
)

# =========================
# DIFFICULTY
# =========================

difficulty_model = joblib.load(
    MODELS_DIR / "difficulty_model.joblib"
)

difficulty_level_model = joblib.load(
    MODELS_DIR / "difficulty_level_model.joblib"
)

# =========================
# FORMAT
# =========================

has_format_model = joblib.load(
    MODELS_DIR / "has_format_model.joblib"
)

format_model = joblib.load(
    MODELS_DIR / "format_model.joblib"
)

format_label_encoder = joblib.load(
    MODELS_DIR / "format_label_encoder.joblib"
)

has_age_model = joblib.load(
    MODELS_DIR / "has_age_model.joblib"
)

age_model = joblib.load(
    MODELS_DIR / "age_model.joblib"
)

has_number_model = joblib.load(
    MODELS_DIR / "has_number_model.joblib"
)
