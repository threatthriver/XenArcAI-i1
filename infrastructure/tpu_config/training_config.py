import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.training import checkpoints
from flax import struct
from typing import Any, Dict, Optional
import optax
import os

@struct.dataclass
class TrainingConfig:
    """Configuration for training on TPU pods."""
    learning_rate: float = 1e-4
    warmup_steps: int = 10000
    total_steps: int = 1000000
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    num_train_epochs: int = 1
    per_device_batch_size: int = 8
    checkpoint_steps: int = 1000
    eval_steps: int = 500
    log_steps: int = 100
    seed: int = 42
    dtype: str = 'bfloat16'
    
    # TPU specific configurations
    num_tpu_cores: int = 256  # For a TPU v3-256 pod
    tpu_zone: str = 'us-central1-a'
    tpu_topology: str = '8x8x8'  # 3D torus network topology
    precision: str = 'bfloat16'
    
    def create_learning_rate_schedule(self):
        """Create learning rate schedule with warmup."""
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=self.learning_rate,
            transition_steps=self.warmup_steps
        )
        decay_fn = optax.cosine_decay_schedule(
            init_value=self.learning_rate,
            decay_steps=self.total_steps - self.warmup_steps
        )
        return optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[self.warmup_steps]
        )

class TPUTrainer:
    def __init__(self, config: TrainingConfig, model_dir: str):
        self.config = config
        self.model_dir = model_dir
        
        # Set up dtype
        self.dtype = jnp.bfloat16 if config.dtype == 'bfloat16' else jnp.float32
        
        # Initialize TPU system
        coordinator_address = os.environ.get('JAX_COORDINATOR_ADDRESS')
        if not coordinator_address:
            # For local development/testing, use CPU/GPU
            self.num_devices = jax.local_device_count()
            return
        
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            process_id=int(os.environ.get('JAX_PROCESS_ID', '0')),
            num_processes=int(os.environ.get('JAX_NUM_PROCESSES', '1')),
            local_device_ids=None  # Auto-detect local devices
        )
        self.num_devices = jax.device_count()
        self.global_batch_size = self.config.per_device_batch_size * self.num_devices
        
        # Set up dtype
        self.dtype = jnp.bfloat16 if config.dtype == 'bfloat16' else jnp.float32
    
    def create_state(self, model, input_shape):
        """Initialize model state across TPU cores."""
        # Create dummy input with correct dtype for embedding
        dummy_input = jnp.ones(input_shape, dtype=jnp.int32)
        
        # Initialize parameters
        variables = model.init(jax.random.PRNGKey(self.config.seed), dummy_input)
        
        # Create optimizer
        lr_schedule = self.config.create_learning_rate_schedule()
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.gradient_clip_norm),
            optax.adamw(
                learning_rate=lr_schedule,
                b1=self.config.beta1,
                b2=self.config.beta2,
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay
            )
        )
        
        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optimizer
        )
    
    def save_checkpoint(self, state, step: int):
        """Save model checkpoint."""
        checkpoints.save_checkpoint(
            ckpt_dir=self.model_dir,
            target=state,
            step=step,
            keep=3
        )
    
    def load_checkpoint(self, state):
        """Load model checkpoint."""
        return checkpoints.restore_checkpoint(
            ckpt_dir=self.model_dir,
            target=state
        )
    
    @staticmethod
    def create_train_step(loss_fn):
        """Create training step function."""
        @jax.jit
        def train_step(state, batch):
            def compute_loss(params):
                logits = state.apply_fn({'params': params}, batch['input_ids'])
                loss = loss_fn(logits, batch['labels'])
                return loss
            
            grad_fn = jax.value_and_grad(compute_loss)
            loss, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            
            return state, loss
        
        return train_step
    
    def create_parallel_train_step(self, train_step):
        """Create parallel training step for TPU pods."""
        return jax.pmap(
            train_step,
            axis_name='batch',
            devices=jax.devices(),
            donate_argnums=(0,)
        )
    
    def create_input_pipeline(self, dataset):
        """Create sharded input pipeline for TPU pods."""
        def shard_batches(batch):
            return jax.tree_map(
                lambda x: x.reshape((self.num_devices, -1) + x.shape[1:]),
                batch
            )
        
        return map(shard_batches, dataset)
    
    def initialize_replicated_state(self, state):
        """Initialize replicated state across TPU cores."""
        return jax.device_put_replicated(
            state,
            jax.devices()
        )