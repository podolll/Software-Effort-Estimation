import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib

keras.utils.set_random_seed(42)
np.random.seed(42)

def cross_validate_keras(build_fn, X, y, n_splits=5, epochs=100, batch_size=32):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values
        model = build_fn(X.shape[1])
        early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stop], verbose=0
        )
        y_pred = model.predict(X_val).flatten()
        mae = mean_absolute_error(y_val, y_pred)
        mae_scores.append(mae)
        keras.backend.clear_session()
    return np.mean(mae_scores), np.std(mae_scores)

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
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='mae',
    )
    return model

def get_sklearn_models():
    return {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    }

def evaluate_sklearn(models, X_tr, X_te, y_tr, y_te):
    results = {}
    for name, m in models.items():
        # Кросс-валидация: сохраняем все значения MAE (в логарифмической шкале)
        cv_scores = -cross_val_score(
            m, X_tr, y_tr, cv=5,
            scoring='neg_mean_absolute_error'
        )
        cv_mae_mean = cv_scores.mean()
        cv_mae_std = cv_scores.std()

        # Обучение и тест
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)

        # Обратное логарифмирование
        y_true = np.exp(y_te)
        y_hat = np.exp(y_pred)

        mae = mean_absolute_error(y_true, y_hat)
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        r2 = r2_score(y_true, y_hat)

        # Сохраняем результаты
        results[name] = {
            'CV_MAE_Mean': cv_mae_mean,
            'CV_MAE_Std': cv_mae_std,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }

        print(f"{name}: CV MAE = {cv_mae_mean:.3f} ± {cv_mae_std:.3f}, "
              f"Test MAE = {mae:.3f}, RMSE = {rmse:.3f}, R² = {r2:.3f}")
    return results

def train_and_evaluate_keras(X_train_scaled, y_train, X_test_scaled, y_test):
    keras_cv_mae, keras_cv_std = cross_validate_keras(
        build_keras_model, X_train_scaled, y_train
    )
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

    # График обучения
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

    return keras_model, history

def save_models(scaler, features, top_proj, top_sub, top_assignee,
                linear_model, rf_model, keras_model,
                path_prefix="DiplomSIP/models/"):
    import os
    os.makedirs(path_prefix, exist_ok=True)

    joblib.dump(linear_model, f"{path_prefix}linear_regression.pkl")
    joblib.dump(rf_model, f"{path_prefix}random_forest.pkl")
    keras_model.save(f"{path_prefix}mlp_model.keras")
    joblib.dump(scaler, f"{path_prefix}scaler.pkl")
    joblib.dump(features, f"{path_prefix}features.pkl")
    joblib.dump(top_proj, f"{path_prefix}top_proj.pkl")
    joblib.dump(top_sub, f"{path_prefix}top_sub.pkl")
    joblib.dump(top_assignee, f"{path_prefix}top_assignee.pkl")
