from pathlib import Path
import joblib
import torch
from app.models.rubert_two_head_model import RuBertTwoHeadFearModel

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
# CATEGORY
# =========================

category_model = joblib.load(
    MODELS_DIR / "category_model.joblib"
)

category_names = joblib.load(
    MODELS_DIR / "category_names.joblib"
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
# =========================
# RUBERT
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUBERT_MODEL_DIR = MODELS_DIR / "rubert_two_head_fear"
RUBERT_MODEL_PATH = RUBERT_MODEL_DIR / "rubert_two_head_fear.pt"

rubert_model = RuBertTwoHeadFearModel(
    model_name=str(RUBERT_MODEL_DIR)
)

state_dict = torch.load(
    RUBERT_MODEL_PATH,
    map_location=DEVICE
)

rubert_model.load_state_dict(state_dict)

rubert_model.to(DEVICE)
rubert_model.eval()