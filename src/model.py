import torch
import torch.nn as nn
from config.config import Config
import math

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

class XenArcModel(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.pos_encoder = PositionalEncoding(config.embedding_dim, config.dropout, config.context_length)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.embedding_dim, nhead=8, dropout=config.dropout),
            num_layers=6  # Increased number of layers
        )
        self.linear = nn.Linear(config.embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.config.embedding_dim) # Scale embeddings
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    config = Config()
    # Dummy vocab_size for testing
    model = XenArcModel(config, vocab_size=10000)
    print(model)
