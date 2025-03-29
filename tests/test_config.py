import unittest
import jax
import jax.numpy as jnp
from model.architecture.transformer import XenArcAI
from infrastructure.tpu_config.training_config import TrainingConfig, TPUTrainer

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.model_config = {
            'vocab_size': 50257,
            'num_layers': 2,
            'num_heads': 8,
            'head_dim': 64,
            'mlp_dim': 512,
            'max_seq_len': 1024,
            'dropout_rate': 0.1,
            'num_experts': 4,
            'expert_dim': 256
        }
        self.training_config = TrainingConfig(
            learning_rate=1e-4,
            warmup_steps=100,
            total_steps=1000,
            per_device_batch_size=8,
            num_train_epochs=1
        )
        self.model = XenArcAI(**self.model_config)
        self.trainer = TPUTrainer(config=self.training_config, model_dir='/tmp/test_model')

    def test_tpu_initialization(self):
        # Test TPU device initialization
        devices = jax.devices()
        self.assertGreater(len(devices), 0)
        
        # Verify device type
        device_type = devices[0].platform
        self.assertIn(device_type.lower(), ['tpu', 'cpu', 'gpu'])

    def test_model_state_initialization(self):
        # Test model state initialization
        input_shape = (self.training_config.per_device_batch_size, self.model_config['max_seq_len'])
        
        # Create dummy input with correct dtype
        dummy_input = jnp.ones(input_shape, dtype=jnp.int32)
        
        # Initialize model state
        variables = self.model.init(jax.random.PRNGKey(0), dummy_input)
        
        # Verify model parameters
        self.assertIn('params', variables)
        
        # Test forward pass
        output = self.model.apply(variables, dummy_input)
        expected_shape = (self.training_config.per_device_batch_size, self.model_config['max_seq_len'], self.model_config['vocab_size'])
        self.assertEqual(output.shape, expected_shape)

    def test_training_config(self):
        # Test training configuration
        self.assertEqual(self.training_config.learning_rate, 1e-4)
        self.assertEqual(self.training_config.warmup_steps, 100)
        self.assertEqual(self.training_config.total_steps, 1000)
        
        # Test learning rate schedule
        schedule = self.training_config.create_learning_rate_schedule()
        warmup_lr = schedule(0)
        peak_lr = schedule(self.training_config.warmup_steps)
        
        self.assertEqual(warmup_lr, 0.0)
        self.assertAlmostEqual(peak_lr, self.training_config.learning_rate)

    def test_trainer_initialization(self):
        # Test trainer initialization
        self.assertIsNotNone(self.trainer.config)
        self.assertIsNotNone(self.trainer.model_dir)
        
        # Test dtype setup
        self.assertEqual(self.trainer.dtype, jnp.bfloat16)
        
        # Test batch size calculation
        expected_global_batch_size = self.training_config.per_device_batch_size * jax.device_count()
        self.assertEqual(self.trainer.global_batch_size, expected_global_batch_size)

if __name__ == '__main__':
    unittest.main()