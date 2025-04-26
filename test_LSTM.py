import torch
import numpy as np

# ======= Класс модели =======
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# ======= Загрузка модели =======
input_size = 8
hidden_size = 64
output_size = 8

model = LSTMModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('lstm_multitask_water_quality_model.pth'))
model.eval()

# ======= Пример ввода =======
example_input = "18.5, 20.0, 2.0, 0.8, 7.2, 8.0, 3.5, 0.3"
input_data = np.array([float(x.strip()) for x in example_input.split(",")])

# Приводим к нужной форме (batch_size, sequence_length, input_size)
input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# ======= Предсказание =======
with torch.no_grad():
    output = model(input_tensor)

predicted_values = output.squeeze().numpy()
print("Предсказанные значения:", predicted_values)
