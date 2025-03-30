import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ActivationStats:
    layer_name: str
    mean: float
    std: float
    sparsity: float

@dataclass
class FunctionUsageStats:
    function_name: str
    call_count: int
    avg_activation: float
    importance_score: float

class ModelAnalyzer:
    def __init__(self, model):
        self.model = model
        self.activation_hooks = {}
        self.layer_stats = {}
        self.function_stats = {}
        self.attention_patterns = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks for all layers to collect activation statistics."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                self.activation_hooks[name] = module.register_forward_hook(
                    lambda m, i, o, name=name: self._activation_hook(name, o)
                )

    def _activation_hook(self, name: str, output):
        """Collect activation statistics for a layer."""
        if isinstance(output, tuple):
            output = output[0]
        
        # Calculate statistics
        with torch.no_grad():
            mean = output.mean().item()
            std = output.std().item()
            sparsity = (output == 0).float().mean().item()

        self.layer_stats[name] = ActivationStats(
            layer_name=name,
            mean=mean,
            std=std,
            sparsity=sparsity
        )

    def analyze_generation(self, input_ids: torch.Tensor) -> Dict:
        """Analyze model's generation behavior."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids)
            
        analysis = {
            'layer_activations': self.layer_stats,
            'function_usage': self._analyze_function_usage(),
            'attention_patterns': self._analyze_attention_patterns(outputs)
        }
        return analysis

    def _analyze_function_usage(self) -> Dict[str, FunctionUsageStats]:
        """Analyze which functions are most active and important."""
        function_stats = {}
        for name, stats in self.layer_stats.items():
            if 'linear' in name.lower() or 'attention' in name.lower():
                importance_score = abs(stats.mean) * (1 - stats.sparsity)
                function_stats[name] = FunctionUsageStats(
                    function_name=name,
                    call_count=1,  # Increment this across multiple calls
                    avg_activation=stats.mean,
                    importance_score=importance_score
                )
        return function_stats

    def _analyze_attention_patterns(self, outputs) -> Dict:
        """Analyze attention patterns in the model."""
        attention_patterns = {}
        if hasattr(outputs, 'attentions'):
            for layer_idx, attention in enumerate(outputs.attentions):
                # Calculate attention statistics
                attention_mean = attention.mean(dim=(0, 1)).cpu().numpy()
                attention_patterns[f'layer_{layer_idx}'] = {
                    'mean_attention': attention_mean,
                    'top_k_tokens': self._get_top_k_attention(attention)
                }
        return attention_patterns

    def _get_top_k_attention(self, attention, k: int = 5) -> List[int]:
        """Get indices of tokens with highest attention scores."""
        mean_attention = attention.mean(dim=(0, 1))
        top_k = torch.topk(mean_attention, k=min(k, mean_attention.size(0)))
        return top_k.indices.cpu().tolist()

    def get_analysis_summary(self) -> Dict:
        """Generate a summary of model analysis."""
        return {
            'total_layers': len(self.layer_stats),
            'active_functions': len(self.function_stats),
            'avg_layer_activation': np.mean([s.mean for s in self.layer_stats.values()]),
            'avg_sparsity': np.mean([s.sparsity for s in self.layer_stats.values()]),
            'most_important_functions': sorted(
                self.function_stats.items(),
                key=lambda x: x[1].importance_score,
                reverse=True
            )[:5]
        }

    def reset_stats(self):
        """Reset all collected statistics."""
        self.layer_stats.clear()
        self.function_stats.clear()
        self.attention_patterns.clear()