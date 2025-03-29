import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from src.model import XenArcModel
from src.data_pipeline import TextDataset
from config.config import Config
import os

def train(model, dataloader, optimizer, epochs, device, vocab_size):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss = None  # Initialize loss variable
    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
        if loss is not None:
            print(f"Epoch {epoch+1}: Loss = {loss.item()}")
        # Save the model checkpoint
        os.makedirs(config.model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.model_dir, f"model_epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TextDataset("data", config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    vocab_size = dataset.tokenizer.vocab_size
    model = XenArcModel(config, vocab_size).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    epochs = 10
    train(model, dataloader, optimizer, epochs, device, vocab_size)
