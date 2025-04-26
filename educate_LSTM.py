import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('water_quality_dataset_big.csv')

features = ['temp_water', 'temp_air', 'precipitation', 'water_level', 'pH', 'oxygen', 'nitrates', 'ammonia']
targets = ['turbidity', 'oxygen', 'water_level'] 

X = df[features].values
y = df[targets].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

class WaterDataset(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = []
        self.y = []
        for i in range(seq_length, len(X)):
            self.X.append(X[i-seq_length:i])
            self.y.append(y[i])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = WaterDataset(X_train, y_train, SEQUENCE_LENGTH)
test_dataset = WaterDataset(X_test, y_test, SEQUENCE_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  
        out = self.fc(out)
        return out

model = LSTMModel(input_size=len(features), hidden_size=64, num_layers=2, output_size=len(targets)).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.6f}')

torch.save(model.state_dict(), 'lstm_multitask_water_quality_model.pth')

print('✅ Модель LSTM многозадачная обучена и сохранена как lstm_multitask_water_quality_model.pth')
