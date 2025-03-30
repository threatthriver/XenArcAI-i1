import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from src.model import XenArcModel
import time
from src.data_pipeline import TextDataset
from config.config import Config
import os

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, device, vocab_size, config):
    """
    Train the XenArcModel model with gradient clipping, validation, and learning rate scheduling.

    Args:
        model (nn.Module): The XenArcModel model.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): Optimizer for training.
        scheduler (LRScheduler): Learning rate scheduler.
        epochs (int): Number of training epochs.
        device (torch.device): Device to train on (CPU or CUDA).
        vocab_size (int): Vocabulary size.
        config (Config): Configuration object.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    for epoch in range(epochs):
        for i, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()

            start_time = time.time()
            output = model(batch)
            forward_time = time.time() - start_time

            loss = criterion(output.view(-1, vocab_size), batch.view(-1))

            start_time = time.time()
            loss.backward()
            backward_time = time.time() - start_time

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping
            optimizer.step()

            print(f"Forward pass time: {forward_time:.4f}s, Backward pass time: {backward_time:.4f}s")
            running_loss += loss.item()

            if i % 100 == 99:
                print(f"Epoch {epoch+1}, Batch {i+1}: Loss = {running_loss / 100:.4f}")
                running_loss = 0.0

        # Validation loop
        val_loss = validate(model, val_dataloader, device, vocab_size, criterion)
        scheduler.step(val_loss) # Step the scheduler with validation loss

        # Save model checkpoint after each epoch
        os.makedirs(config.model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.model_dir, f"model_epoch_{epoch+1}.pth"))
        print(f"Epoch {epoch+1} finished, validation loss: {val_loss:.4f}, model saved.")

    # Check if the device supports quantization
    if torch.backends.quantized.engine == 'none':
        print("Quantization is not supported on this device")
    else:
        # Quantize the model
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

def validate(model, dataloader, device, vocab_size, criterion):
    """
    Validate the model on the given dataloader.

    Args:
        model (nn.Module): The XenArcModel model.
        dataloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to validate on (CPU or CUDA).
        vocab_size (int): Vocabulary size.
        criterion (Loss): Loss function.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output.view(-1, vocab_size), batch.view(-1))
            running_loss += loss.item()
    model.train() # Set back to train mode
    return running_loss / len(dataloader)

if __name__ == '__main__':
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TextDataset("data", config)

    # Split data into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    vocab_size = dataset.tokenizer.vocab_size
    model = XenArcModel(config, vocab_size).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) # Learning rate scheduler
    epochs = config.num_epochs
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, device, vocab_size, config)
