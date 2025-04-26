import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('water_quality_dataset.csv')

def classify_quality(turbidity):
    if turbidity > 15:
        return 'bad'
    elif turbidity > 8:
        return 'medium'
    else:
        return 'good'

df['quality_class'] = df['turbidity'].apply(classify_quality)

features = ['temp_water', 'temp_air', 'precipitation', 'water_level', 'pH', 'oxygen', 'nitrates', 'ammonia']

X = df[features]
y = df['quality_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    loss_function='MultiClass',
    verbose=100
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

model.save_model('catboost_model.cbm')

print("✅ CatBoost модель обучена и сохранена как catboost_model.cbm")
