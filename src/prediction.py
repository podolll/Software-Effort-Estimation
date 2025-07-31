import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Пути к моделям и артефактам (относительно этого файла)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../models"))

model = load_model(os.path.join(MODEL_DIR, "mlp_model.keras"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
top_proj = joblib.load(os.path.join(MODEL_DIR, "top_proj.pkl"))
top_sub = joblib.load(os.path.join(MODEL_DIR, "top_sub.pkl"))
top_assignee = joblib.load(os.path.join(MODEL_DIR, "top_assignee.pkl"))

def predict_from_input(input_dict):
    # Преобразуем вход в DataFrame
    df_input = pd.DataFrame([input_dict])

    # Логарифмируем оценку - используем np.log, как в обучении (не log1p)
    df_input["LogHoursEstimate"] = np.log(df_input["HoursEstimate"])

    # Обработка редких категорий
    df_input["ProjGroup"] = df_input["ProjectCode"].apply(
        lambda x: x if x in top_proj else "Other_ProjectCode")
    df_input["SubGroup"] = df_input["SubCategory"].apply(
        lambda x: x if x in top_sub else "Other_SubCategory")
    df_input["AssigneeGroup"] = df_input["AssignedToID"].apply(
        lambda x: str(x) if x in top_assignee else "Other_AssigneeGroup")

    # One-hot кодирование
    df_encoded = pd.get_dummies(df_input, columns=["ProjGroup", "SubGroup", "AssigneeGroup"])
    for f in features:
        if f not in df_encoded.columns:
            df_encoded[f] = 0
    df_encoded = df_encoded[features]

    # Стандартизация числовых признаков
    num_features = ["LogHoursEstimate", "Priority"]
    cat_features = [f for f in features if f not in num_features]

    X_num_scaled = scaler.transform(df_encoded[num_features])
    X_input = np.hstack([X_num_scaled, df_encoded[cat_features].values])

    # Прогноз и обратное преобразование
    y_pred_log = model.predict(X_input).flatten()[0]
    y_pred_real = np.exp(y_pred_log)  # обратное к np.log

    return y_pred_real
