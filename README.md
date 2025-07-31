# Software-Effort-Estimation
# Дипломная работа: Предсказание трудозатрат в разработке ПО с использованием ML

Этот репозиторий содержит материалы по моей выпускной квалификационной работе на тему:
**"Применение методов машинного обучения для коррекции оценок трудозатрат в задачах разработки ПО"**

# Структура репозитория
```
📁 Software-Effort-Estimation/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/ # Набор данных
│ ├── README.md
│ ├── Sip-task_info.csv
│
├── models/ # Сохранённые модели и артефакты после обучения
│ ├── features.pkl
│ ├── linear_regression.pkl
│ ├── mlp_model.keras
│ ├── random_forest.pkl
│ ├── scaler.pkl
│ ├── top_assignee.pkl
│ ├── top_proj.pkl
│ ├── top_sub.pkl
│
├── src/
│ ├── data_processing.py # Загрузка и подготовка данных
│ ├── model_training.py # Обучение моделей, оценка, сохранение
│ ├── prediction.py # Загрузка моделей, функция предсказания
│ ├── interface.py # Streamlit веб-интерфейс для прогноза
│ ├── main.py # Скрипт запуска обучения и сохранения моделей
│
├── figures/ # Графики
│ ├── eda/ # графики из части исследовательского анализа данных
│ ├── results/ # графики полученных результатов


```
