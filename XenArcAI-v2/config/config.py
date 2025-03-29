import flax.struct

@flax.struct.dataclass
class ModelConfig:
    """Configuration for the XenArcAI model architecture."""
    vocab_size: int = 50257  # Example: GPT-2 tokenizer size
    num_layers: int = 12      # Reduced for faster testing/dev
    num_heads: int = 12
    head_dim: int = 64
    mlp_dim: int = 3072       # Typically 4 * num_heads * head_dim
    max_seq_len: int = 1024   # Reduced context window for testing
    dropout_rate: float = 0.1
    num_experts: int = 4      # Reduced MoE for testing
    expert_dim: int = 512     # Reduced MoE for testing
    dtype: str = 'float32'    # Data type (e.g., 'float32', 'bfloat16')

@flax.struct.dataclass
class DataConfig:
    """Configuration for the data pipeline."""
    tokenizer_name: str = 'gpt2'
    max_seq_length: int = 1024 # Should match model's max_seq_len
    batch_size_per_device: int = 8
    num_workers: int = 4
    shuffle_buffer_size: int = 1000
    # Placeholder for dataset paths/URLs
    train_data_pattern: str = "data_placeholder/train-*.tfrecord" # Example pattern
    eval_data_pattern: str = "data_placeholder/eval-*.tfrecord"   # Example pattern

@flax.struct.dataclass
class TrainingConfig:
    """Configuration for the training process."""
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    warmup_steps: int = 1000
    log_steps: int = 100
    checkpoint_steps: int = 500
    eval_steps: int = 500
    model_dir: str = "checkpoints"
    log_dir: str = "logs"
    seed: int = 42

@flax.struct.dataclass
class AppConfig:
    """Main configuration container."""
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()

# Function to load/get config (can be extended later for YAML/CLI args)
def get_config() -> AppConfig:
    return AppConfig()
