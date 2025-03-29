import unittest
import jax
import jax.numpy as jnp
from model.architecture.transformer import XenArcAI
from infrastructure.tpu_config.training_config import TrainingConfig, TPUTrainer

class TestTPUTraining(unittest.TestCase):
    def setUp(self):
        # Initialize model and training configurations
        self.model_config = {
            'vocab_size': 50257,
            'num_layers': 2,  # Reduced for testing
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

    def test_model_initialization(self):
        # Test model initialization on TPU
        input_shape = (self.training_config.per_device_batch_size, self.model_config['max_seq_len'])
        state = self.trainer.create_state(self.model, input_shape)
        
        # Verify state components
        self.assertIsNotNone(state.params)
        self.assertIsNotNone(state.apply_fn)
        self.assertIsNotNone(state.tx)

    def test_training_step(self):
        # Test single training step
        input_shape = (self.training_config.per_device_batch_size, self.model_config['max_seq_len'])
        state = self.trainer.create_state(self.model, input_shape)

        # Create dummy batch
        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(
            key,
            (self.training_config.per_device_batch_size, self.model_config['max_seq_len']),
            0,
            self.model_config['vocab_size']
        )
        batch = {'input_ids': input_ids, 'labels': input_ids}

        # Define loss function
        def cross_entropy_loss(logits, labels):
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))

        # Create and run training step
        train_step = self.trainer.create_train_step(cross_entropy_loss)
        new_state, loss = train_step(state, batch)

        # Verify training step results
        self.assertIsNotNone(loss)
        self.assertLess(loss, float('inf'))
        self.assertGreater(loss, float('-inf'))

    def test_parallel_training(self):
        # Test parallel training on multiple devices
        input_shape = (self.training_config.per_device_batch_size, self.model_config['max_seq_len'])
        state = self.trainer.create_state(self.model, input_shape)

        # Initialize replicated state
        replicated_state = self.trainer.initialize_replicated_state(state)

        # Create dummy batch for all devices
        key = jax.random.PRNGKey(0)
        global_batch_size = self.training_config.per_device_batch_size * jax.device_count()
        input_ids = jax.random.randint(
            key,
            (global_batch_size, self.model_config['max_seq_len']),
            0,
            self.model_config['vocab_size']
        )
        batch = {'input_ids': input_ids, 'labels': input_ids}

        # Prepare batch for parallel training
        sharded_batch = jax.tree_map(
            lambda x: x.reshape((jax.device_count(), -1) + x.shape[1:]),
            batch
        )

        # Define loss function
        def cross_entropy_loss(logits, labels):
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))

        # Create and run parallel training step
        train_step = self.trainer.create_train_step(cross_entropy_loss)
        p_train_step = self.trainer.create_parallel_train_step(train_step)
        new_state, loss = p_train_step(replicated_state, sharded_batch)

        # Verify parallel training results
        self.assertIsNotNone(loss)
        self.assertEqual(loss.shape, (jax.device_count(),))

    def test_checkpointing(self):
        # Test model checkpointing
        input_shape = (self.training_config.per_device_batch_size, self.model_config['max_seq_len'])
        state = self.trainer.create_state(self.model, input_shape)

        # Save checkpoint
        self.trainer.save_checkpoint(state, step=0)

        # Load checkpoint
        loaded_state = self.trainer.load_checkpoint(state)

        # Verify loaded state
        self.assertIsNotNone(loaded_state)
        self.assertEqual(
            jax.tree_util.tree_structure(loaded_state),
            jax.tree_util.tree_structure(state)
        )

if __name__ == '__main__':
    unittest.main()