import numpy as np
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model('catboost_water_quality_classifier')

print("Введите параметры качества воды через пробел:")
print("temp_water (°C), temp_air (°C), precipitation (мм), water_level (м), pH, oxygen (мг/л), nitrates (мг/л), ammonia (мг/л)")
print("Пример: 20.5 15.3 0.0 1.2 7.2 8.1 2.5 0.1")
user_input = input("Введите данные: ")

if not user_input.strip():
    print("Используем пример: 20.5 15.3 0.0 1.2 7.2 8.1 2.5 0.1")
    user_input = "20.5 15.3 0.0 1.2 7.2 8.1 2.5 0.1"

input_values = list(map(float, user_input.strip().split()))

assert len(input_values) == 8, "Нужно ввести ровно 8 признаков!"

X_sample = np.array([input_values])

prediction = model.predict(X_sample)

label_mapping = {0: 'Чистая вода', 1: 'Средняя мутность', 2: 'Сильная мутность'}


print("\n🔍 Оценка мутности воды:")
print(f"Класс: {label_mapping[int(prediction[0])]}")

