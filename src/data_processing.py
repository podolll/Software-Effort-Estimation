import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_process_data(file_path='Software-Effort-Estimation/data/Sip-task-info.csv', encoding='cp1251', random_state=42):
    # Загрузка и очистка
    df = pd.read_csv(file_path, encoding=encoding)
    df = df[(df['HoursActual'] == df['DeveloperHoursActual']) & (df['Category'] == 'Development')]
    df = df[df['AssignedToID'] == df['DeveloperID']]
    df = df.drop(columns=[
        'TaskNumber', 'Summary', 'RaisedByID', 'AuthorisedByID', 'StatusCode',
        'ProjectBreakdownCode', 'Category', 'DeveloperID', 'DeveloperHoursActual',
        'TaskPerformance', 'DeveloperPerformance'
    ])

    # Убираем нулевые и отрицательные значения
    df = df[(df['HoursEstimate'] > 0) & (df['HoursActual'] > 0)].copy()

    # Более мягкое удаление выбросов (99-й перцентиль)
    est_99 = df['HoursEstimate'].quantile(0.99)
    act_99 = df['HoursActual'].quantile(0.99)
    df = df[(df['HoursEstimate'] >= 0.25) &
            (df['HoursEstimate'] <= est_99) &
            (df['HoursActual'] >= 0.25) &
            (df['HoursActual'] <= act_99)].copy()

    # Логарифмы
    df['LogHoursEstimate'] = np.log(df['HoursEstimate'])
    df['LogHoursActual'] = np.log(df['HoursActual'])

    # Проекты
    proj_counts = df['ProjectCode'].value_counts()
    top_proj = proj_counts[proj_counts > 50].index
    df['ProjGroup'] = df['ProjectCode']
    df.loc[~df['ProjectCode'].isin(top_proj), 'ProjGroup'] = 'Other_ProjectCode'

    # Подкатегории
    sub_counts = df['SubCategory'].value_counts()
    top_sub = sub_counts[sub_counts > 50].index
    df['SubGroup'] = df['SubCategory']
    df.loc[~df['SubCategory'].isin(top_sub), 'SubGroup'] = 'Other_SubCategory'

    # Исполнители
    assignee_counts = df['AssignedToID'].value_counts()
    top_assignee = assignee_counts[assignee_counts > 50].index
    df['AssigneeGroup'] = df['AssignedToID'].astype(str)
    df.loc[~df['AssignedToID'].isin(top_assignee), 'AssigneeGroup'] = 'Other_AssigneeGroup'

    # Сплит на train/test
    X_raw = df.copy()
    y_raw = df['LogHoursActual']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw,
        test_size=0.2,
        random_state=random_state,
        stratify=X_raw['ProjGroup']
    )

    # One-hot кодирование
    categorical_features = ['AssigneeGroup', 'ProjGroup', 'SubGroup']
    X_train_ohe = pd.get_dummies(X_train_raw, columns=categorical_features, drop_first=True)
    X_test_ohe = pd.get_dummies(X_test_raw, columns=categorical_features, drop_first=True)
    X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)

    # Отбор признаков
    categorical_prefixes = ['AssigneeGroup_', 'ProjGroup_', 'SubGroup_']
    num_features = ['LogHoursEstimate', 'Priority']
    cat_features = [c for c in X_train_ohe.columns if any(c.startswith(p) for p in categorical_prefixes)]

    # Основной (немасштабированный) датасет
    features = num_features + cat_features
    X_train = X_train_ohe[features].copy()
    X_test = X_test_ohe[features].copy()

    # Масштабирование только числовых признаков — для нейросети
    scaler = StandardScaler()
    X_train_scaled_num = scaler.fit_transform(X_train[num_features])
    X_test_scaled_num = scaler.transform(X_test[num_features])

    X_train_scaled = np.hstack([X_train_scaled_num, X_train[cat_features].values])
    X_test_scaled = np.hstack([X_test_scaled_num, X_test[cat_features].values])

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'features': features,
        'top_proj': top_proj.tolist(),
        'top_sub': top_sub.tolist(),
        'top_assignee': top_assignee.tolist()
    }
