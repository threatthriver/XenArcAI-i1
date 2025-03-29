import torch
import torch.nn as nn
from config.config import Config

class XenArcModel(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=config.embedding_dim, nhead=8)
        self.linear = nn.Linear(config.embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    config = Config()
    # Dummy vocab_size for testing
    model = XenArcModel(config, vocab_size=10000)
    print(model)
