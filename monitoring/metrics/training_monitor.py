import wandb
import tensorflow as tf
import jax.numpy as jnp
import psutil
import os
from typing import Dict, Any, Optional
from datetime import datetime

class TrainingMonitor:
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: Dict[str, Any],
        log_dir: str,
        enable_wandb: bool = True
    ):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize W&B
        if enable_wandb:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config
            )
        
        # Initialize TensorBoard
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(log_dir, 'tensorboard')
        )
        
        # Initialize metrics
        self.metrics = {}
        self.step = 0
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training metrics to all enabled platforms."""
        current_step = step if step is not None else self.step
        
        # Update internal metrics
        self.metrics.update(metrics)
        
        # Log to W&B
        wandb.log(metrics, step=current_step)
        
        # Log to TensorBoard
        with self.summary_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=current_step)
        
        self.step = current_step + 1
    
    def log_model_gradients(self, gradients: Dict[str, jnp.ndarray], step: Optional[int] = None):
        """Log gradient statistics during training."""
        grad_stats = {}
        for param_name, grad in gradients.items():
            grad_stats.update({
                f'gradients/{param_name}/mean': float(jnp.mean(grad)),
                f'gradients/{param_name}/std': float(jnp.std(grad)),
                f'gradients/{param_name}/norm': float(jnp.linalg.norm(grad))
            })
        
        self.log_metrics(grad_stats, step)
    
    def log_resource_usage(self, step: Optional[int] = None):
        """Log system resource utilization."""
        resources = {
            'system/cpu_percent': psutil.cpu_percent(),
            'system/memory_percent': psutil.virtual_memory().percent,
            'system/disk_usage_percent': psutil.disk_usage('/').percent
        }
        
        self.log_metrics(resources, step)
    
    def log_tpu_metrics(self, step: Optional[int] = None):
        """Log TPU-specific metrics."""
        # This is a placeholder for TPU metrics
        # Actual implementation would depend on TPU monitoring API
        pass
    
    def log_attention_patterns(self, attention_weights: jnp.ndarray, step: Optional[int] = None):
        """Log attention visualization."""
        # Log attention heatmap
        fig = wandb.Image(attention_weights)
        wandb.log({"attention_patterns": fig}, step=step)
    
    def log_expert_usage(self, expert_counts: Dict[str, int], step: Optional[int] = None):
        """Log MoE expert utilization."""
        expert_metrics = {
            f'moe/expert_{i}_count': count 
            for i, count in enumerate(expert_counts)
        }
        expert_metrics['moe/expert_utilization'] = sum(expert_counts.values()) / len(expert_counts)
        
        self.log_metrics(expert_metrics, step)
    
    def log_memory_usage(self, step: Optional[int] = None):
        """Log memory usage statistics."""
        memory = psutil.virtual_memory()
        memory_metrics = {
            'memory/total_gb': memory.total / (1024 ** 3),
            'memory/available_gb': memory.available / (1024 ** 3),
            'memory/used_gb': memory.used / (1024 ** 3),
            'memory/percent': memory.percent
        }
        
        self.log_metrics(memory_metrics, step)
    
    def save_model_checkpoint(self, model_state: Any, step: int):
        """Log model checkpoint to W&B."""
        checkpoint_path = os.path.join(self.log_dir, f'checkpoint-{step}')
        wandb.save(checkpoint_path)
    
    def finish(self):
        """Cleanup and finalize logging."""
        wandb.finish()
        self.summary_writer.close()