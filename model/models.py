

import torch
import torch.nn as nn
import math
class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerEncoderModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):

        src = self.input_linear(src) * math.sqrt(self.hidden_dim)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):


        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class WeakDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=1):
        super(WeakDecoder, self).__init__()

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)


        output = self.linear(lstm_out)

        return output


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
