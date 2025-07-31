import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Загрузка всех артефактов
model = load_model("models/mlp_model.keras")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")
top_proj = joblib.load("models/top_proj.pkl")
top_sub = joblib.load("models/top_sub.pkl")
top_assignee = joblib.load("models/top_assignee.pkl")

def predict_from_input(input_dict):
    # Шаг 1: Преобразуем вход в DataFrame
    df_input = pd.DataFrame([input_dict])

    # Шаг 2: Логарифмируем оценку
    df_input["LogHoursEstimate"] = np.log1p(df_input["HoursEstimate"])

    # Шаг 3: Обработка редких категорий
    df_input["ProjGroup"] = df_input["ProjectCode"].apply(
        lambda x: x if x in top_proj else "Other_ProjectCode")
    df_input["SubGroup"] = df_input["SubCategory"].apply(
        lambda x: x if x in top_sub else "Other_SubCategory")
    df_input["AssigneeGroup"] = df_input["AssignedToID"].apply(
        lambda x: str(x) if x in top_assignee else "Other_AssigneeGroup")

    # Шаг 4: One-hot кодирование
    df_encoded = pd.get_dummies(df_input, columns=["ProjGroup", "SubGroup", "AssigneeGroup"])
    for f in features:
        if f not in df_encoded.columns:
            df_encoded[f] = 0
    df_encoded = df_encoded[features]

    # Шаг 5: Стандартизация числовых признаков
    num_features = ["LogHoursEstimate", "Priority"]
    cat_features = [f for f in features if f not in num_features]

    X_num_scaled = scaler.transform(df_encoded[num_features])
    X_input = np.hstack([X_num_scaled, df_encoded[cat_features].values])

    # Шаг 6: Прогноз и обратное преобразование
    y_pred_log = model.predict(X_input).flatten()[0]
    y_pred_real = np.expm1(y_pred_log)
    return y_pred_real
