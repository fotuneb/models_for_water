import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Параметры
SEQ_LENGTH = 7
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001

df = pd.read_csv('water_quality_dataset.csv')

df['turbidity_next_day'] = df.groupby('lake_id')['turbidity'].shift(-1)
df = df.dropna()

features = ['temp_water', 'temp_air', 'precipitation', 'water_level', 'pH', 'oxygen', 'nitrates', 'ammonia']

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

X = []
y = []

for lake_id in df['lake_id'].unique():
    df_lake = df[df['lake_id'] == lake_id]
    for i in range(len(df_lake) - SEQ_LENGTH):
        X.append(df_lake[features].iloc[i:i+SEQ_LENGTH].values)
        y.append(df_lake['turbidity_next_day'].iloc[i+SEQ_LENGTH])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()

model = LSTMRegressor(input_size=X_train.shape[2], hidden_size=64, num_layers=2)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Обучение
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), 'lstm_model.pth')

print("✅ LSTM модель обучена и сохранена как lstm_model.pth")
