import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib

# Установка сидов для воспроизводимости
keras.utils.set_random_seed(42)
np.random.seed(42)

# === Загрузка и очистка данных ===
df = pd.read_csv('Sip/Sip-task-info.csv', encoding='cp1251')
df = df[(df['HoursActual'] == df['DeveloperHoursActual']) & (df['Category'] == 'Development')]
df = df[df['AssignedToID'] == df['DeveloperID']]
df = df.drop(columns=[
    'TaskNumber', 'Summary', 'RaisedByID', 'AuthorisedByID', 'StatusCode',
    'ProjectBreakdownCode', 'Category', 'DeveloperID', 'DeveloperHoursActual',
    'TaskPerformance', 'DeveloperPerformance'
])
df = df[(df['HoursEstimate'] > 0) & (df['HoursActual'] > 0)].copy()

# Удаление выбросов
est_99 = df['HoursEstimate'].quantile(0.99)
act_99 = df['HoursActual'].quantile(0.99)
df = df[(df['HoursEstimate'] >= 0.25) & (df['HoursEstimate'] <= est_99) &
        (df['HoursActual'] >= 0.25) & (df['HoursActual'] <= act_99)].copy()

# Логарифмирование
df['LogHoursEstimate'] = np.log(df['HoursEstimate'])
df['LogHoursActual'] = np.log(df['HoursActual'])

# Обработка категориальных признаков
proj_counts = df['ProjectCode'].value_counts()
top_proj = proj_counts[proj_counts > 50].index
df['ProjGroup'] = df['ProjectCode']
df.loc[~df['ProjectCode'].isin(top_proj), 'ProjGroup'] = 'Other_ProjectCode'

sub_counts = df['SubCategory'].value_counts()
top_sub = sub_counts[sub_counts > 50].index
df['SubGroup'] = df['SubCategory']
df.loc[~df['SubCategory'].isin(top_sub), 'SubGroup'] = 'Other_SubCategory'

assignee_counts = df['AssignedToID'].value_counts()
top_assignee = assignee_counts[assignee_counts > 50].index
df['AssigneeGroup'] = df['AssignedToID'].astype(str)
df.loc[~df['AssignedToID'].isin(top_assignee), 'AssigneeGroup'] = 'Other_AssigneeGroup'

# Разделение на train/test
X_raw = df.copy()
y_raw = df['LogHoursActual']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=X_raw['ProjGroup']
)

# One-hot кодирование
categorical_features = ['AssigneeGroup', 'ProjGroup', 'SubGroup']
X_train_ohe = pd.get_dummies(X_train_raw, columns=categorical_features, drop_first=True)
X_test_ohe = pd.get_dummies(X_test_raw, columns=categorical_features, drop_first=True)
X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)

# Выделение признаков
categorical_prefixes = ['AssigneeGroup_', 'ProjGroup_', 'SubGroup_']
num_features = ['LogHoursEstimate', 'Priority']
cat_features = [c for c in X_train_ohe.columns if any(c.startswith(p) for p in categorical_prefixes)]
features = num_features + cat_features
X_train = X_train_ohe[features].copy()
X_test = X_test_ohe[features].copy()

# Масштабирование числовых признаков
scaler = StandardScaler()
X_train_scaled_num = scaler.fit_transform(X_train[num_features])
X_test_scaled_num = scaler.transform(X_test[num_features])
X_train_scaled = np.hstack([X_train_scaled_num, X_train[cat_features].values])
X_test_scaled = np.hstack([X_test_scaled_num, X_test[cat_features].values])

# === Модель Keras ===
def build_keras_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mae')
    return model

def cross_validate_keras(build_fn, X, y, n_splits=5, epochs=100, batch_size=32):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values
        model = build_fn(X.shape[1])
        early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=0)
        y_pred = model.predict(X_val).flatten()
        mae = mean_absolute_error(y_val, y_pred)
        mae_scores.append(mae)
        keras.backend.clear_session()
    return np.mean(mae_scores), np.std(mae_scores)

keras_cv_mae, keras_cv_std = cross_validate_keras(build_keras_model, X_train_scaled, y_train)
print(f"Keras MLP CV MAE = {keras_cv_mae:.3f} ± {keras_cv_std:.3f}")

keras_model = build_keras_model(X_train_scaled.shape[1])
early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history = keras_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2, epochs=200, batch_size=32,
    callbacks=[early_stop], verbose=0
)
y_pred_log = keras_model.predict(X_test_scaled).flatten()
y_test_real = np.exp(y_test)
y_pred_real = np.exp(y_pred_log)
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
r2 = r2_score(y_test_real, y_pred_real)
print(f"Keras MLP Test MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

# === sklearn модели ===
def get_sklearn_models():
    return {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    }

def evaluate_sklearn(models, X_tr, X_te, y_tr, y_te):
    results = {}
    for name, m in models.items():
        cv_scores = -cross_val_score(m, X_tr, y_tr, cv=5, scoring='neg_mean_absolute_error')
        cv_mae_mean = cv_scores.mean()
        cv_mae_std = cv_scores.std()
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        y_true = np.exp(y_te)
        y_hat = np.exp(y_pred)
        mae = mean_absolute_error(y_true, y_hat)
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        r2 = r2_score(y_true, y_hat)
        results[name] = {
            'CV_MAE_Mean': cv_mae_mean,
            'CV_MAE_Std': cv_mae_std,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        print(f"{name}: CV MAE = {cv_mae_mean:.3f} ± {cv_mae_std:.3f}, Test MAE = {mae:.3f}, RMSE = {rmse:.3f}, R² = {r2:.3f}")
    return results

sk_models = get_sklearn_models()
sk_results = evaluate_sklearn(sk_models, X_train, X_test, y_train, y_test)

# === Визуализация ===
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='MAE на обучении')
plt.plot(history.history['val_loss'], label='MAE на валидации')
plt.xlabel('Эпоха')
plt.ylabel('Средняя абсолютная ошибка (MAE)')
plt.title('Динамика обучения нейронной сети')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_curve_mlp.png")
plt.show()

# === Сохранение моделей и артефактов ===
joblib.dump(sk_models['LinearRegression'], "models/linear_regression.pkl")
joblib.dump(sk_models['RandomForest'], "models/random_forest.pkl")
keras_model.save("models/mlp_model.keras")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(features, "models/features.pkl")
joblib.dump(top_proj.tolist(), "models/top_proj.pkl")
joblib.dump(top_sub.tolist(), "models/top_sub.pkl")
joblib.dump(top_assignee.tolist(), "models/top_assignee.pkl")
