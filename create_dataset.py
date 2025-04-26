import pandas as pd
import numpy as np

# Параметры
np.random.seed(42)
n_days = 365
n_lakes = 5

# Даты
dates = pd.date_range(start='2024-01-01', periods=n_days)

# Список данных
data = []

for lake_id in range(1, n_lakes + 1):
    temp_water = []
    temp_air = []
    precipitation = np.random.exponential(scale=2, size=n_days)
    water_level = np.random.normal(loc=2, scale=0.3, size=n_days)
    
    # Генерация сезонной температуры
    for date in dates:
        month = date.month
        if month in [12, 1, 2]:
            tw = np.random.normal(loc=4, scale=2)
            ta = tw + np.random.normal(loc=1, scale=2)
        elif month in [6, 7, 8]:
            tw = np.random.normal(loc=20, scale=3)
            ta = tw + np.random.normal(loc=2, scale=2)
        else:
            tw = np.random.normal(loc=12, scale=3)
            ta = tw + np.random.normal(loc=1, scale=2)
        
        temp_water.append(tw)
        temp_air.append(ta)
    
    temp_water = np.array(temp_water)
    temp_air = np.array(temp_air)

    # Показатели воды
    pH = np.clip(np.random.normal(loc=7, scale=0.5, size=n_days), 6, 8.5)
    turbidity = np.abs(np.random.normal(loc=5, scale=2, size=n_days))
    oxygen = np.clip(np.random.normal(loc=8, scale=1, size=n_days), 5, 12)
    nitrates = np.abs(np.random.normal(loc=1, scale=0.5, size=n_days))
    ammonia = np.abs(np.random.normal(loc=0.2, scale=0.1, size=n_days))
    
    # Аномалии + сезонные коррекции
    for i in range(n_days):
        month = dates[i].month

        # Сильные осадки
        if precipitation[i] > 10:
            turbidity[i] += np.random.uniform(5, 10)  # Рост мутности
            oxygen[i] -= np.random.uniform(1, 2)      # Падение кислорода
            oxygen[i] = max(oxygen[i], 0)
            nitrates[i] += np.random.uniform(0.5, 1.5)  # Рост нитратов
        
        # Летние коррекции
        if month in [6, 7, 8]:
            pH[i] += np.random.uniform(0.1, 0.3)  # Повышение pH за счет фотосинтеза
            pH[i] = min(pH[i], 8.5)               # Ограничение сверху

        data.append([
            dates[i], lake_id, temp_water[i], temp_air[i], precipitation[i],
            water_level[i], pH[i], turbidity[i], oxygen[i], nitrates[i], ammonia[i]
        ])

# Создание DataFrame
columns = ['date', 'lake_id', 'temp_water', 'temp_air', 'precipitation',
           'water_level', 'pH', 'turbidity', 'oxygen', 'nitrates', 'ammonia']

df = pd.DataFrame(data, columns=columns)

# Сохраняем в CSV
df.to_csv('water_quality_dataset.csv', index=False)

print("✅ Датасет успешно сохранён как 'water_quality_dataset.csv'")
print(df.head(10))
