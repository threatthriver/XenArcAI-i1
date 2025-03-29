import unittest
import jax
import jax.numpy as jnp

# Adjust import path based on how you run tests (e.g., from the root XenArcAI-v2 dir)
# If running with `python -m unittest discover tests`, this should work.
from src.model import XenArcAI
from config.config import get_config, ModelConfig

class TestModel(unittest.TestCase):

    def test_model_initialization_and_forward_pass(self):
        """Tests if the model can be initialized and perform a forward pass."""
        config = get_config()
        model_config = config.model
        key = jax.random.PRNGKey(0)
        model = XenArcAI(config=model_config)

        # Create dummy input
        batch_size = 2
        seq_len = model_config.max_seq_len # Use configured max length
        dummy_input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        print(f"\n--- Testing Model Initialization ---")
        print(f"Model Config: {model_config}")
        print(f"Input shape: {dummy_input_ids.shape}")

        try:
            params = model.init(key, dummy_input_ids, deterministic=True)['params']
            print("Model initialized successfully.")

            # Test forward pass
            print("\n--- Testing Model Forward Pass ---")
            logits = model.apply({'params': params}, dummy_input_ids, deterministic=True)
            print(f"Output logits shape: {logits.shape}")

            # Check output shape
            expected_shape = (batch_size, seq_len, model_config.vocab_size)
            self.assertEqual(logits.shape, expected_shape)
            print("Forward pass successful, output shape is correct.")

        except Exception as e:
            self.fail(f"Model initialization or forward pass failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
