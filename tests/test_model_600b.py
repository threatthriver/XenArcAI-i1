import os
import sys
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent))

from src.model import XenArcModel
from src.model_analysis import ModelAnalyzer

@dataclass
class ModelConfig:
    embedding_dim: int
    dropout: float
    context_length: int
    vocab_size: int
    num_heads: int
    num_layers: int

    @classmethod
    def from_dict(cls, config: Dict) -> 'ModelConfig':
        model_config = config['model']
        return cls(
            embedding_dim=model_config['hidden_size'],
            dropout=model_config['dropout_rate'],
            context_length=model_config['max_position_embeddings'],
            vocab_size=model_config['vocab_size'],
            num_heads=model_config['num_attention_heads'],
            num_layers=model_config['num_hidden_layers']
        )

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ModelTester:
    def __init__(self, config_path: str):
        self.config_dict = load_config(config_path)
        self.config = ModelConfig.from_dict(self.config_dict)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._initialize_model()
        self.analyzer = ModelAnalyzer(self.model)

    def _initialize_model(self) -> XenArcModel:
        model = XenArcModel(self.config, self.config.vocab_size).to(self.device)
        return model

    def _generate_test_input(self, batch_size: int = 32) -> torch.Tensor:
        return torch.randint(
            0, self.config.vocab_size,
            (batch_size, self.config.context_length),
            device=self.device
        )

    def run_analysis(self, batch_size: int = 32) -> None:
        print(f"\nModel Architecture:")
        print(f"- Embedding Dimension: {self.config.embedding_dim}")
        print(f"- Number of Layers: {self.config.num_layers}")
        print(f"- Number of Attention Heads: {self.config.num_heads}")
        print(f"- Context Length: {self.config.context_length}")
        print(f"- Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        input_ids = self._generate_test_input(batch_size)
        analysis = self.analyzer.analyze_generation(input_ids)

        self._print_analysis_results(analysis)

    def _print_analysis_results(self, analysis: Dict) -> None:
        print("\nModel Analysis Results:")
        
        # Layer Statistics
        layer_stats = analysis['layer_activations']
        print("\nLayer Statistics:")
        for name, stats in layer_stats.items():
            print(f"- {name}:")
            print(f"  Mean Activation: {stats.mean:.4f}")
            print(f"  Activation Std: {stats.std:.4f}")
            print(f"  Sparsity: {stats.sparsity:.4f}")

        # Function Usage
        print("\nMost Active Functions:")
        for name, stats in analysis['function_usage'].items():
            print(f"- {name}: Score = {stats.importance_score:.4f}")

        # Attention Patterns
        if analysis['attention_patterns']:
            print("\nAttention Analysis:")
            for layer, pattern in analysis['attention_patterns'].items():
                print(f"\n{layer}:")
                print(f"Top attended positions: {pattern['top_k_tokens']}")

def test_model_generation():
    tester = ModelTester('config/model_600b.yaml')
    tester.run_analysis()

if __name__ == '__main__':
    test_model_generation()
    print("\nModel testing completed successfully!")