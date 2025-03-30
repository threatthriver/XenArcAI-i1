import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import XenArcModel
from data_pipeline import TextDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_optimizer(model, config):
    initial_lr = config['training']['learning_rate']['initial']
    min_lr = config['training']['learning_rate']['min']
    max_steps = config['training']['max_steps']
    
    optimizer = Adam(model.parameters(), lr=initial_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=min_lr)
    
    return optimizer, scheduler

def train_step(model, optimizer, inputs, labels, config):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(inputs)
    loss = nn.CrossEntropyLoss()(predictions.view(-1, predictions.size(-1)), labels.view(-1))
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clipping'])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def validate(model, dataset, steps):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(dataset):
            if step >= steps:
                break
            predictions = model(inputs)
            loss = criterion(predictions.view(-1, predictions.size(-1)), labels.view(-1))
            total_loss += loss.item()
    
    return total_loss / steps

def main():
    # Load configuration
    config = load_config('config/tpu_config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Enable mixed precision if configured
    scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
    
    # Create datasets
    train_dataset = TextDataset(config['data']['train_path'], config)
    eval_dataset = TextDataset(config['data']['eval_path'], config)
    
    # Create model and optimizer
    model = XenArcModel(config['model'], config['model']['vocab_size']).to(device)
    optimizer, scheduler = create_optimizer(model, config)
    
    # Setup checkpointing
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    for step in range(config['training']['max_steps']):
        # Training step
        batch = next(iter(train_dataset))
        inputs, labels = [b.to(device) for b in batch]
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = train_step(model, optimizer, inputs, labels, config)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = train_step(model, optimizer, inputs, labels, config)
        
        scheduler.step()
        
        # Logging
        if step % config['logging']['log_every_n_steps'] == 0:
            print(f"Step {step}: loss = {loss:.4f}")
        
        # Validation
        if step % config['validation']['steps'] == 0:
            val_loss = validate(model, eval_dataset, config['validation']['steps'])
            print(f"Validation loss: {val_loss:.4f}")
        
        # Checkpointing
        if step % config['logging']['save_checkpoints_steps'] == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': step,
            }
            torch.save(checkpoint, f'checkpoints/checkpoint_{step}.pt')

if __name__ == '__main__':
    main()