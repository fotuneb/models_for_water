import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('datasets/water_dataset.csv')

features = ['oxygen', 'pH', 'nitrates', 'ammonia', 'precipitation', 'water_level', 'turbidity']
target = 'quality_class'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    verbose=50,
    random_seed=42
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model.save_model('water_catboost_model.cbm')
print("\n✅ Модель сохранена как 'water_catboost_model.cbm'")
