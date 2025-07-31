from src.data_processing import load_and_process_data
from src.model_training import (
    get_sklearn_models, evaluate_sklearn,
    train_and_evaluate_keras, save_models
)

def main():
    print("Загрузка и обработка данных...")
    data = load_and_process_data()

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    scaler = data['scaler']
    features = data['features']
    top_proj = data['top_proj']
    top_sub = data['top_sub']
    top_assignee = data['top_assignee']

    print("Обучение sklearn моделей...")
    sk_models = get_sklearn_models()
    sk_results = evaluate_sklearn(sk_models, X_train, X_test, y_train, y_test)

    print("Обучение нейронной сети Keras...")
    keras_model, history = train_and_evaluate_keras(X_train_scaled, y_train, X_test_scaled, y_test)

    print("Сохранение моделей и артефактов...")
    save_models(
        scaler=scaler,
        features=features,
        top_proj=top_proj,
        top_sub=top_sub,
        top_assignee=top_assignee,
        linear_model=sk_models['LinearRegression'],
        rf_model=sk_models['RandomForest'],
        keras_model=keras_model
    )
    print("Обучение завершено, модели сохранены.")

if __name__ == "__main__":
    main()
