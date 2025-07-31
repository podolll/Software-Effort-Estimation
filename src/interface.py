import streamlit as st
from prediction import predict_from_input

st.set_page_config(page_title="Коррекция оценки задачи", layout="centered")

st.title("💡 Коррекция оценки задачи разработки ПО")

st.markdown("Введите параметры задачи, чтобы получить откорректированную моделью оценку времени выполнения.")

with st.form("input_form"):
    hours_estimate = st.number_input("🔢 Первичная оценка (часов)", min_value=0.25, step=0.25)
    priority = st.slider("⭐ Приоритет задачи", min_value=1, max_value=10, value=3)
    
    project_code = st.text_input("📁 Проект", value="PRJ001")
    subcategory = st.text_input("🧩 Подкатегория задачи", value="Feature")
    assigned_to = st.text_input("👤 ID исполнителя", value="23")

    submitted = st.form_submit_button("📤 Получить прогноз")

if submitted:
    try:
        # Базовая валидация
        if hours_estimate <= 0.25:
            st.error("Оценка должна быть больше 0.25.")
        else:
            input_data = {
                "HoursEstimate": hours_estimate,
                "Priority": priority,
                "ProjectCode": project_code,
                "SubCategory": subcategory,
                "AssignedToID": int(assigned_to)
            }

            pred_hours = predict_from_input(input_data)
            
            st.markdown(f"⏱️ Прогнозируемое время выполнения: **{pred_hours:.2f} ч.**")

            # Анализ расхождения и индикация риска
            ratio = pred_hours / hours_estimate

            # Градация риска
            if ratio < 0.7 or ratio > 1.5:
                st.error("🔴 **Высокий уровень риска**: значительное расхождение между оценкой и прогнозом. Рекомендуется пересмотреть задачу.")
            else:
                st.success("🟢 **Низкий уровень риска**: система подтверждает пользовательскую оценку.")

    except ValueError:
        st.error("Невозможно обработать входные данные. Проверьте правильность ID исполнителя и других параметров.")
