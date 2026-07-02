import re
from pathlib import Path

import joblib


# =========================
# PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGE_MODEL_PATH = PROJECT_ROOT / "app" / "models" / "age_model.joblib"


# =========================
# LOAD MODEL
# =========================

age_bundle = joblib.load(AGE_MODEL_PATH)

# Новый вариант: сохранен словарь {"model": model, ...}
if isinstance(age_bundle, dict):
    age_model = age_bundle["model"]

# Старый вариант: сохранена сразу модель
else:
    age_model = age_bundle


# =========================
# AGE RULES
# =========================

def normalize_text(text: str) -> str:
    return str(text).lower().strip()


def age_number_to_group(age):
    age = int(age)

    if age <= 7:
        return "6+"
    elif age <= 9:
        return "8+"
    elif age <= 11:
        return "10+"
    elif age <= 13:
        return "12+"
    elif age <= 15:
        return "14+"
    elif age <= 17:
        return "16+"
    else:
        return "18+"


def is_people_context_after(text: str, end: int) -> bool:
    """
    Проверяем, что число относится к количеству людей, а не к возрасту.

    Например:
    8-10 чел
    8-10 человек
    8-10 игроков
    """

    window_after = text[end:end + 25]

    people_patterns = [
        r"\s*(чел|человек|человека|игроков|игрока|участников|персон|ребят|ребята)\b",
    ]

    return any(
        re.match(pattern, window_after)
        for pattern in people_patterns
    )


def extract_age_rule(text):
    text = normalize_text(text)
    text = text.replace("ё", "е")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    plus_match = re.search(
        r"\b(6|8|10|12|14|16|18)\s*\+",
        text
    )

    if plus_match:
        return f"{plus_match.group(1)}+"

    age_list_match = re.search(
        r"\b((?:\d{1,2}\s*(?:,|и|/|\s)\s*)+\d{1,2})\s*"
        r"(?:лет|года|год|г\.?)\b",
        text
    )

    if age_list_match:
        age_numbers = re.findall(
            r"\d{1,2}",
            age_list_match.group(1)
        )

        ages = [
            int(age)
            for age in age_numbers
            if 1 <= int(age) <= 100
        ]

        if ages:
            return age_number_to_group(min(ages))

    # =========================
    # 1. Возрастной диапазон с обязательным словом "лет"
    # 30-35 лет, 10-12 лет
    # =========================

    range_with_age_word_match = re.search(
        r"\b(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*"
        r"(?:лет|года|год|г\.?)\b",
        text
    )

    if range_with_age_word_match:
        age_1 = int(range_with_age_word_match.group(1))
        age_2 = int(range_with_age_word_match.group(2))

        min_age = min(age_1, age_2)

        return age_number_to_group(min_age)

    # =========================
    # 2. Возрастной диапазон с контекстом перед ним
    # возраст 30-35
    # возрастная категория 30-35
    # категория 30-35
    # =========================

    range_with_context_match = re.search(
        r"\b(?:возраст|возраста|возрастная\s+категория|категория)\s*"
        r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\b",
        text
    )

    if range_with_context_match:
        age_1 = int(range_with_context_match.group(1))
        age_2 = int(range_with_context_match.group(2))

        min_age = min(age_1, age_2)

        return age_number_to_group(min_age)

    # =========================
    # 3. Диапазоны словами:
    # от 10 до 12 лет
    # с 8 до 10 лет
    #
    # Важно: слово "лет" здесь лучше требовать,
    # иначе "от 4 до 7 человек" может стать возрастом.
    # =========================

    word_range_match = re.search(
        r"\b(?:от|с)\s*(\d{1,2})\s*до\s*(\d{1,2})\s*"
        r"(?:лет|года|год|г\.?)\b",
        text
    )

    if word_range_match:
        age_1 = int(word_range_match.group(1))
        age_2 = int(word_range_match.group(2))

        min_age = min(age_1, age_2)

        return age_number_to_group(min_age)

    age_matches = re.findall(
        r"\b(\d{1,2})\s*(?:лет|года|год|годик|годика|годиков|г\.?)\b",
        text
    )

    if age_matches:
        ages = [
            int(x)
            for x in age_matches
            if 1 <= int(x) <= 100
        ]

        if ages:
            return age_number_to_group(min(ages))

    age_context_match = re.search(
        r"\b(?:"
        r"возраст|возраста|"
        r"ребенку|ребенок|ребёнку|ребёнок|"
        r"дочке|сыну|"
        r"детям|ребятам|"
        r"девочке|мальчику|"
        r"девушка|парень|подросток|подростку"
        r")\s*(?:от|с)?\s*(\d{1,2})\b",
        text
    )

    if age_context_match:
        return age_number_to_group(int(age_context_match.group(1)))

    from_age_match = re.search(
        r"\b(?:от|с)\s*(\d{1,2})\s*(?:лет|года|год|г\.?)\b",
        text
    )

    if from_age_match:
        return age_number_to_group(int(from_age_match.group(1)))

    return None


# =========================
# PREDICT
# =========================

def predict_age(text):
    """
    Главная функция предсказания возраста.

    Сначала применяем правила.
    Если правило нашло возраст — возвращаем его.
    Если нет — используем ML-модель.
    """

    text = normalize_text(text)

    rule_age = extract_age_rule(text)

    if rule_age is not None:
        return {
            "has_age": 1,
            "age": rule_age,
            "source": "rule"
        }

    pred_age = age_model.predict([text])[0]

    return {
        "has_age": 1,
        "age": pred_age,
        "source": "model"
    }
