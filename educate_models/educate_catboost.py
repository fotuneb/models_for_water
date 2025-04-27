import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
import optuna

# Загрузка данных
df = pd.read_csv('datasets/water_quality_dataset_big.csv')

# Маппинг класса качества
quality_mapping = {'good': 0, 'medium': 1, 'bad': 2}
df['quality_label'] = df['quality_class'].map(quality_mapping)

features = ['temp_water', 'temp_air', 'precipitation', 'water_level', 'pH', 'oxygen', 'nitrates', 'ammonia']
target = 'quality_label'

X = df[features].values
y = df[target].values

# Разделение данных
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Определение функции оптимизации
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 3000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'MultiClass',
        'verbose': 0,
        'task_type': 'CPU'  # Можно поставить 'GPU', если есть видеокарта
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, preds)
    return accuracy

# Запуск оптимизации
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print('✅ Лучшие параметры:', study.best_params)
print(f'🎯 Лучший результат: {study.best_value:.4f}')

best_params = study.best_params
best_params['loss_function'] = 'MultiClass'
best_params['verbose'] = 50

final_model = CatBoostClassifier(**best_params)
final_model.fit(X_train, y_train)

# Оценка на тестовых данных
y_pred = final_model.predict(X_valid)
print('Финальная точность:', accuracy_score(y_valid, y_pred))
print('📋 Classification Report:')
print(classification_report(y_valid, y_pred, target_names=['good', 'medium', 'bad']))

final_model.save_model('models/catboost_water_quality_classifier_optimized')
print('✅ Оптимизированная модель сохранена!')
