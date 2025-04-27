import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('dataset/water_quality_dataset_big.csv')

def label_quality(turbidity):
    if turbidity < 5:
        return 0
    elif turbidity <= 15:
        return 1
    else:
        return 2

df['quality_label'] = df['turbidity'].apply(label_quality)

features = ['temp_water', 'temp_air', 'precipitation', 'water_level', 'pH', 'oxygen', 'nitrates', 'ammonia']
target = 'quality_label'

X = df[features].values
y = df[target].values

scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.01,
    loss_function='MultiClass',
    verbose=50,
    class_weights=[3, 1, 3]  
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

model.save_model('models/catboost_water_quality_classifier')

print('✅ Классификатор CatBoost обучен и сохранён как catboost_water_quality_classifier')
