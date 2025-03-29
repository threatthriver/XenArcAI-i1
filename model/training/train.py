import jax
import jax.numpy as jnp
from typing import Dict, Optional
import os
import logging
from tqdm import tqdm

from model.architecture.transformer import XenArcAI
from data.pipeline.data_pipeline import WebDataPipeline
from infrastructure.tpu_config.training_config import TrainingConfig, TPUTrainer

def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_model_config():
    return {
        'vocab_size': 50257,  # GPT-2 vocabulary size
        'num_layers': 96,      # Deep architecture for 600B params
        'num_heads': 96,
        'head_dim': 128,
        'mlp_dim': 24576,
        'max_seq_len': 100000,  # Extended context window
        'dropout_rate': 0.1,
        'num_experts': 8,
        'expert_dim': 1024
    }

def create_loss_fn():
    def loss_fn(logits, labels):
        # Cross entropy loss
        one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
        loss = -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1)
        return jnp.mean(loss)
    return loss_fn

def main():
    # Setup directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'checkpoints')
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir)
    
    # Initialize model and training config
    model_config = create_model_config()
    training_config = TrainingConfig()
    
    # Create model instance
    model = XenArcAI(**model_config)
    
    # Initialize TPU trainer
    trainer = TPUTrainer(training_config, model_dir)
    
    # Initialize data pipeline
    data_pipeline = WebDataPipeline(
        tokenizer_name='gpt2',  # Using GPT-2 tokenizer
        max_seq_length=model_config['max_seq_len'],
        batch_size=training_config.per_device_batch_size
    )
    
    # Create training dataset
    train_dataset = data_pipeline.create_jax_dataset(
        file_pattern=os.path.join(base_dir, 'data/processed/train-*.tar')
    )
    
    # Create evaluation dataset
    eval_dataset = data_pipeline.create_jax_dataset(
        file_pattern=os.path.join(base_dir, 'data/processed/eval-*.tar')
    )
    
    # Initialize model state
    input_shape = (training_config.per_device_batch_size, model_config['max_seq_len'])
    state = trainer.create_state(model, input_shape)
    
    # Load checkpoint if exists
    state = trainer.load_checkpoint(state) or state
    
    # Create training step
    train_step = trainer.create_train_step(create_loss_fn())
    p_train_step = trainer.create_parallel_train_step(train_step)
    
    # Initialize replicated state
    state = trainer.initialize_replicated_state(state)
    
    # Training loop
    logger.info('Starting training...')
    for epoch in range(training_config.num_train_epochs):
        # Training
        for step, batch in enumerate(tqdm(train_dataset, desc=f'Epoch {epoch}')):
            # Training step (batch is already sharded by data pipeline)
            state, loss = p_train_step(state, batch)
            
            # Logging
            if step % training_config.log_steps == 0:
                logger.info(f'Step {step}, Loss: {loss}')
            
            # Save checkpoint
            if step % training_config.checkpoint_steps == 0:
                trainer.save_checkpoint(state, step)
            
            # Evaluation
            if step % training_config.eval_steps == 0:
                eval_losses = []
                for eval_batch in eval_dataset:
                    _, eval_loss = p_train_step(state, eval_batch)
                    eval_losses.append(eval_loss)
                avg_eval_loss = jnp.mean(jnp.array(eval_losses))
                logger.info(f'Step {step}, Eval Loss: {avg_eval_loss}')
    
    logger.info('Training completed!')

if __name__ == '__main__':
    main()