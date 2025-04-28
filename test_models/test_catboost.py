import numpy as np
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model('models/water_catboost_model.cbm')

label_mapping = {0: 'Чистая вода', 1: 'Средняя мутность', 2: 'Сильная мутность'}

print("Введите параметры качества воды через пробел:")
print("oxygen (мг/л), pH, nitrates (мг/л), ammonia (мг/л), precipitation (мм), water_level (м), turbidity (NTU)")
print("Пример: 8.1 7.2 2.5 0.1 0.0 1.2 10")

user_input = input("Введите данные: ")

if not user_input.strip():
    print("Используем пример: 8.1 7.2 2.5 0.1 0.0 1.2 10 , 8.1 7.2 1.4 0.1 0.0 1.2 5")
    user_input = "8.1 7.2 2.5 0.1 0.0 1.2 10"

try:
    input_values = list(map(float, user_input.strip().split()))
    assert len(input_values) == 7, "Нужно ввести ровно 7 признаков!"
except ValueError:
    print("Ошибка: Все значения должны быть числами.")
    exit()
except AssertionError:
    print("Ошибка: Необходимо ввести 7 значений.")
    exit()

X_sample = np.array([input_values])

prediction = model.predict(X_sample)

print("\n🔍 Оценка качества воды:")
print(f"Класс: {prediction[0]}")
