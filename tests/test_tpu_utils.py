import unittest
import jax
import jax.numpy as jnp
import optax
from model.architecture.transformer import XenArcAI
from infrastructure.tpu_config.training_config import TrainingConfig, TPUTrainer

class TestTPUUtils(unittest.TestCase):
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

    def test_input_pipeline(self):
        # Test input pipeline sharding
        batch_size = self.training_config.per_device_batch_size
        seq_len = self.model_config['max_seq_len']
        
        # Create dummy dataset
        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(key, (batch_size * 4, seq_len), 0, self.model_config['vocab_size'])
        dataset = [{'input_ids': input_ids} for _ in range(5)]
        
        # Test sharding
        sharded_dataset = self.trainer.create_input_pipeline(dataset)
        first_batch = next(iter(sharded_dataset))
        
        # Verify shapes
        expected_shape = (jax.device_count(), -1, seq_len)
        self.assertEqual(first_batch['input_ids'].shape[0], jax.device_count())

    def test_learning_rate_schedule(self):
        # Test learning rate schedule
        schedule = self.training_config.create_learning_rate_schedule()
        
        # Test warmup phase
        warmup_lr = schedule(0)
        self.assertEqual(warmup_lr, 0.0)
        
        mid_warmup_lr = schedule(self.training_config.warmup_steps // 2)
        self.assertGreater(mid_warmup_lr, 0.0)
        self.assertLess(mid_warmup_lr, self.training_config.learning_rate)
        
        peak_lr = schedule(self.training_config.warmup_steps)
        self.assertAlmostEqual(peak_lr, self.training_config.learning_rate)

    def test_gradient_clipping(self):
        # Test gradient clipping in training
        input_shape = (self.training_config.per_device_batch_size, self.model_config['max_seq_len'])
        state = self.trainer.create_state(self.model, input_shape)
        
        # Create dummy batch with large gradients
        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(
            key,
            (self.training_config.per_device_batch_size, self.model_config['max_seq_len']),
            0,
            self.model_config['vocab_size']
        )
        batch = {'input_ids': input_ids, 'labels': input_ids}
        
        # Define loss function that produces large gradients
        def large_gradient_loss(logits, labels):
            return 1e6 * jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        
        # Create and run training step
        train_step = self.trainer.create_train_step(large_gradient_loss)
        new_state, loss = train_step(state, batch)
        
        # Verify loss is finite (not NaN or inf)
        self.assertTrue(jnp.isfinite(loss))

    def test_dtype_conversion(self):
        # Test dtype handling
        input_shape = (self.training_config.per_device_batch_size, self.model_config['max_seq_len'])
        
        # Test with different dtypes
        for dtype in ['float32', 'bfloat16']:
            config = TrainingConfig(
                learning_rate=1e-4,
                warmup_steps=100,
                total_steps=1000,
                per_device_batch_size=8,
                dtype=dtype
            )
            trainer = TPUTrainer(config=config, model_dir='/tmp/test_model')
            
            # Create dummy input
            dummy_input = jnp.ones(input_shape)
            expected_dtype = jnp.bfloat16 if dtype == 'bfloat16' else jnp.float32
            
            # Verify trainer's dtype
            self.assertEqual(trainer.dtype, expected_dtype)

if __name__ == '__main__':
    unittest.main()