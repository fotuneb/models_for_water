import pandas as pd
import numpy as np

n_lakes = 100 
n_days = 365   

np.random.seed(42)

data = []

for lake_id in range(n_lakes):
    for day in range(n_days):
        date = pd.to_datetime('2024-01-01') + pd.Timedelta(days=day)
        temp_water = np.random.normal(loc=15, scale=5)
        temp_air = np.random.normal(loc=10, scale=10)
        precipitation = np.random.exponential(scale=2)
        water_level = np.random.normal(loc=3, scale=1)
        pH = np.random.normal(loc=7, scale=0.5)
        oxygen = np.random.normal(loc=8, scale=2)
        nitrates = np.random.normal(loc=2, scale=1)
        ammonia = np.random.normal(loc=0.5, scale=0.2)
        turbidity = np.random.normal(loc=10, scale=5)  

        temp_water = np.clip(temp_water, 0, 30)
        temp_air = np.clip(temp_air, -10, 40)
        precipitation = np.clip(precipitation, 0, 20)
        water_level = np.clip(water_level, 0, 10)
        pH = np.clip(pH, 5, 9)
        oxygen = np.clip(oxygen, 0, 14)
        nitrates = np.clip(nitrates, 0, 10)
        ammonia = np.clip(ammonia, 0, 2)
        turbidity = np.clip(turbidity, 0, 30)

        data.append([
            lake_id, date, temp_water, temp_air, precipitation, 
            water_level, pH, oxygen, nitrates, ammonia, turbidity
        ])

columns = ['lake_id', 'date', 'temp_water', 'temp_air', 'precipitation', 
           'water_level', 'pH', 'oxygen', 'nitrates', 'ammonia', 'turbidity']

df = pd.DataFrame(data, columns=columns)

def classify_quality(turbidity):
    if turbidity > 15:
        return 'bad'
    elif turbidity > 8:
        return 'medium'
    else:
        return 'good'

df['quality_class'] = df['turbidity'].apply(classify_quality)

print(df['quality_class'].value_counts())

df.to_csv('datasets\water_quality_dataset_big.csv', index=False)

print('✅ Датасет сохранён как water_quality_dataset_big.csv')
