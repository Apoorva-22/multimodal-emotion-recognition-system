import torch
import torch.nn as nn


class AudioEmotionModel(nn.Module):
    def __init__(self):
        super(AudioEmotionModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        # 128D embedding for fusion
        self.fc1 = nn.Linear(128, 128)

        # final classification
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        last_output = lstm_out[:, -1, :]

        embedding = self.fc1(last_output)

        output = self.fc2(embedding)

        return output, embedding