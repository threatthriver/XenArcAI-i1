import unittest
import jax
import jax.numpy as jnp
from model.architecture.transformer import XenArcAI, FlashMultiHeadAttention, MoELayer
from data.pipeline.data_pipeline import WebDataPipeline
from infrastructure.tpu_config.training_config import TrainingConfig, TPUTrainer

class TestXenArcAI(unittest.TestCase):
    def setUp(self):
        self.model_config = {
            'vocab_size': 50257,
            'num_layers': 2,  # Reduced for testing
            'num_heads': 8,
            'head_dim': 64,
            'mlp_dim': 512,
            'max_seq_len': 1024,  # Reduced for testing
            'dropout_rate': 0.1,
            'num_experts': 4,
            'expert_dim': 256
        }
        self.model = XenArcAI(**self.model_config)
        self.batch_size = 2
        self.seq_length = 512
    
    def test_model_forward(self):
        # Test model forward pass
        key = jax.random.PRNGKey(0)
        input_shape = (self.batch_size, self.seq_length)
        input_ids = jax.random.randint(key, input_shape, 0, self.model_config['vocab_size'])
        
        # Initialize parameters
        variables = self.model.init(key, input_ids)
        
        # Forward pass
        output = self.model.apply(variables, input_ids)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_length, self.model_config['vocab_size'])
        self.assertEqual(output.shape, expected_shape)
    
    def test_flash_attention(self):
        # Test Flash Attention module
        attention = FlashMultiHeadAttention(
            num_heads=8,
            head_dim=64,
            dropout_rate=0.1
        )
        
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 16, 512))  # (batch, seq_len, hidden_dim)
        
        # Initialize parameters
        variables = attention.init(key, x)
        
        # Forward pass
        output = attention.apply(variables, x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
    
    def test_moe_layer(self):
        # Test Mixture of Experts layer
        moe = MoELayer(
            num_experts=4,
            expert_dim=256
        )
        
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 16, 512))  # (batch, seq_len, hidden_dim)
        
        # Initialize parameters
        variables = moe.init(key, x)
        
        # Forward pass
        output = moe.apply(variables, x)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 16, 512))

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = WebDataPipeline(
            tokenizer_name='gpt2',
            max_seq_length=1024,
            batch_size=2
        )
    
    def test_preprocess_text(self):
        # Test text preprocessing
        text = "This is a test sentence."
        output = self.pipeline.preprocess_text(text)
        
        # Check output format
        self.assertIn('input_ids', output)
        self.assertIn('attention_mask', output)
        
        # Check shapes
        self.assertEqual(output['input_ids'].shape[1], self.pipeline.max_seq_length)
        self.assertEqual(output['attention_mask'].shape[1], self.pipeline.max_seq_length)
    
    def test_filter_data(self):
        # Test data filtering
        valid_example = {'text': 'This is a valid example with more than ten words to ensure proper length validation.'}
        invalid_example = {'text': 'Short.'}
        empty_example = {'text': ''}
        whitespace_example = {'text': '   \n\t   '}
        non_string_example = {'text': 123}
        
        # Test valid case
        self.assertTrue(self.pipeline.filter_data(valid_example))
        
        # Test invalid cases
        self.assertFalse(self.pipeline.filter_data(invalid_example))
        self.assertFalse(self.pipeline.filter_data(empty_example))
        self.assertFalse(self.pipeline.filter_data(whitespace_example))
        self.assertFalse(self.pipeline.filter_data(non_string_example))

class TestTPUTrainer(unittest.TestCase):
    def setUp(self):
        self.config = TrainingConfig()
        self.model_dir = '/tmp/test_model'
        self.trainer = TPUTrainer(self.config, self.model_dir)
    
    def test_create_state(self):
        # Test model state creation
        model = XenArcAI(
            vocab_size=50257,
            num_layers=2,
            num_heads=8,
            head_dim=64,
            mlp_dim=512
        )
        
        input_shape = (self.config.per_device_batch_size, 1024)
        state = self.trainer.create_state(model, input_shape)
        
        # Check state attributes
        self.assertIsNotNone(state.params)
        self.assertIsNotNone(state.apply_fn)
        self.assertIsNotNone(state.tx)
    
    def test_create_train_step(self):
        # Test training step creation
        def dummy_loss_fn(logits, labels):
            return jnp.mean((logits - labels) ** 2)
        
        train_step = self.trainer.create_train_step(dummy_loss_fn)
        self.assertIsNotNone(train_step)

if __name__ == '__main__':
    unittest.main()