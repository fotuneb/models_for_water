import pandas as pd
import numpy as np

np.random.seed(42)

n_days = 365
n_lakes = 100
dates = pd.date_range(start='2024-01-01', periods=n_days)

data = []

for lake_id in range(1, n_lakes + 1):
    temp_water = []
    temp_air = []
    precipitation = np.random.exponential(scale=2, size=n_days)
    water_level = np.clip(np.random.normal(loc=2, scale=0.5, size=n_days), 0, 10)
    
    # Сезонные температуры
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
        
        temp_water.append(np.clip(tw, 0, 30))
        temp_air.append(np.clip(ta, -10, 40))
    
    temp_water = np.array(temp_water)
    temp_air = np.array(temp_air)

    pH = np.clip(np.random.normal(loc=7, scale=0.5, size=n_days), 6, 8.5)
    turbidity = np.clip(np.abs(np.random.normal(loc=5, scale=2, size=n_days)), 0, 30)
    oxygen = np.clip(np.random.normal(loc=8, scale=1, size=n_days), 5, 14)
    nitrates = np.clip(np.abs(np.random.normal(loc=1, scale=0.5, size=n_days)), 0, 10)
    ammonia = np.clip(np.abs(np.random.normal(loc=0.2, scale=0.1, size=n_days)), 0, 2)
    
    for i in range(n_days):
        month = dates[i].month
        
        # Аномалии при осадках
        if precipitation[i] > 10:
            turbidity[i] += np.random.uniform(5, 10)
            oxygen[i] = max(oxygen[i] - np.random.uniform(1, 2), 0)
            nitrates[i] += np.random.uniform(0.5, 1.5)
        
        # Летние изменения
        if month in [6, 7, 8]:
            pH[i] += np.random.uniform(0.1, 0.3)
            pH[i] = min(pH[i], 8.5)

        data.append([
            lake_id, dates[i], temp_water[i], temp_air[i], precipitation[i],
            water_level[i], pH[i], oxygen[i], nitrates[i], ammonia[i], turbidity[i]
        ])

columns = ['lake_id', 'date', 'temp_water', 'temp_air', 'precipitation', 
           'water_level', 'pH', 'oxygen', 'nitrates', 'ammonia', 'turbidity']

df = pd.DataFrame(data, columns=columns)

def classify_quality(row):
    """
    Новая логика качества воды:
    - 'bad', если мутность высокая ИЛИ кислорода мало ИЛИ азот высокий тд
    - 'medium', если всё среднее
    - 'good', если вода чистая по всем метрикам
    """
    score = 0
    
    if row['oxygen'] >= 6:
        score += 1
    if 6.8 <= row['pH'] <= 8.2:
        score += 1
    if row['nitrates'] <= 1.5:
        score += 1
    if row['ammonia'] <= 0.4:
        score += 1
    if row['precipitation'] <= 4:
        score += 1
    if 2 <= row['water_level'] <= 4:
        score += 1
    if row['turbidity'] <= 8:
        score += 1

    if score >= 7:
        return 'good'
    elif 5 <= score <= 6:
        return 'medium'
    else:
        return 'bad'

df['quality_class'] = df.apply(classify_quality, axis=1)

df.to_csv('datasets/water_dataset.csv', index=False)

print("✅ Датасет успешно сохранён как 'datasets/water_dataset.csv'")
print(df['quality_class'].value_counts())
