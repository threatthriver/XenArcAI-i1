import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Tuple
import functools

class FlashMultiHeadAttention(nn.Module):
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project input into queries, keys, and values
        qkv = nn.Dense(3 * self.num_heads * self.head_dim, dtype=self.dtype)(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, batch, heads, seq, dim)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        
        # Flash attention computation
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attention = jnp.einsum('bhsd,bhtd->bhst', queries, keys)
        attention = jnp.multiply(attention, scale)
        attention = jax.nn.softmax(attention, axis=-1)
        
        if training and self.dropout_rate > 0:
            attention = nn.dropout(attention, rate=self.dropout_rate)
        
        output = jnp.einsum('bhst,bhtd->bhsd', attention, values)
        output = jnp.transpose(output, (0, 2, 1, 3))  # (batch, seq, heads, dim)
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Final projection
        output = nn.Dense(self.num_heads * self.head_dim, dtype=self.dtype)(output)
        
        return output

class MoELayer(nn.Module):
    num_experts: int
    expert_dim: int
    capacity_factor: float = 1.0
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Router
        router_weights = self.param('router_weights',
                                  nn.initializers.xavier_uniform(),
                                  (hidden_dim, self.num_experts))
        
        # Calculate routing probabilities
        route_logits = jnp.einsum('bsh,he->bse', x, router_weights)
        routing_weights = jax.nn.softmax(route_logits)
        
        # Expert computation
        expert_outputs = []
        for i in range(self.num_experts):
            expert = nn.Dense(
                features=self.expert_dim,
                dtype=self.dtype,
                kernel_init=nn.initializers.xavier_uniform()
            )
            expert_output = expert(x)
            expert_outputs.append(expert_output)
        
        # Combine expert outputs
        expert_outputs = jnp.stack(expert_outputs, axis=-2)
        combined_output = jnp.einsum('bse,bseh->bsh', routing_weights, expert_outputs)
        
        # Final projection back to input dimension
        output = nn.Dense(
            features=hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform()
        )(combined_output)
        
        return output
        
        # Select top-k experts
        k = max(1, int(self.num_experts * self.capacity_factor))
        top_k_weights, top_k_indices = jax.lax.top_k(routing_weights, k)
        
        # Expert computation
        expert_outputs = []
        for i in range(self.num_experts):
            expert = nn.Dense(self.expert_dim, dtype=self.dtype)
            # Create binary mask for expert selection
            mask = jnp.equal(top_k_indices, i)[..., None]
            # Expand mask to match input dimensions
            mask = jnp.broadcast_to(mask, x.shape)
            # Apply mask and compute expert output
            expert_input = jnp.where(mask, x, 0.)
            expert_output = expert(expert_input)
            # Apply routing weight
            weight = jnp.expand_dims(top_k_weights[..., i], axis=-1)
            expert_outputs.append(expert_output * weight)
        
        # Combine expert outputs
        combined_output = sum(expert_outputs)
        return combined_output
        return combined_output

class TransformerBlock(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int
    dropout_rate: float = 0.1
    num_experts: int = 8
    expert_dim: int = 1024
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Layer normalization 1
        y = nn.LayerNorm(dtype=self.dtype)(x)
        
        # Flash Multi-head attention
        y = FlashMultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate)(y, training=not deterministic)
        
        # Residual connection 1
        x = x + y
        
        # Layer normalization 2
        y = nn.LayerNorm(dtype=self.dtype)(x)
        
        # MoE layer
        y = MoELayer(
            num_experts=self.num_experts,
            expert_dim=self.expert_dim,
            dtype=self.dtype)(y)
        
        # Residual connection 2
        x = x + y
        
        return x
        
        # Layer normalization 2
        y = nn.LayerNorm(dtype=self.dtype)(x)
        
        # MoE FFN
        y = MoELayer(
            num_experts=self.num_experts,
            expert_dim=hidden_dim,  # Match input dimension
            dtype=self.dtype)(y)
        
        # Residual connection 2
        return x + y

class XenArcAI(nn.Module):
    vocab_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    mlp_dim: int
    max_seq_len: int
    dropout_rate: float = 0.1
    num_experts: int = 8
    expert_dim: int = 1024
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, input_ids):
        # Ensure input_ids are integers
        input_ids = input_ids.astype(jnp.int32)
        
        # Embedding layer
        x = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.mlp_dim,
            dtype=self.dtype,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )(input_ids)
        
        # Add positional embeddings
        positions = jnp.arange(input_ids.shape[1])[None]
        position_embedding = nn.Embed(
            num_embeddings=self.max_seq_len,
            features=self.mlp_dim,
            dtype=self.dtype
        )(positions)
        # Broadcast position embedding to match input shape
        position_embedding = jnp.broadcast_to(
            position_embedding,
            (input_ids.shape[0], input_ids.shape[1], self.mlp_dim)
        )
        x = x + position_embedding
        
        # Apply dropout
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        
        # Transformer layers
        for _ in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                num_experts=self.num_experts,
                expert_dim=self.expert_dim,
                dtype=self.dtype
            )(x)
        
        # Final layer normalization
        x = nn.LayerNorm(dtype=self.dtype)(x)
        
        # Output projection
        logits = nn.Dense(
            features=self.vocab_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )(x)
        
        return logits
    num_layers: int
    num_heads: int
    head_dim: int
    mlp_dim: int
    max_seq_len: int = 100000  # Extended context window
    dropout_rate: float = 0.1
    num_experts: int = 8
    expert_dim: int = 1024
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True):
        # Token embedding
        # Ensure input_ids are integers
        input_ids = input_ids.astype(jnp.int32)
        x = nn.Embed(self.vocab_size, self.mlp_dim, dtype=self.dtype)(input_ids)
        
        # Rotary position embeddings
        positions = jnp.arange(input_ids.shape[1])[None, :]
        freqs = 1.0 / (10000 ** (2 * jnp.arange(self.mlp_dim // 2) / self.mlp_dim))
        sinusoids = positions[:, :, None] * freqs[None, None, :]
        rotary_pos_emb = jnp.concatenate([jnp.sin(sinusoids), jnp.cos(sinusoids)], axis=-1)
        rotary_pos_emb = jnp.broadcast_to(rotary_pos_emb, x.shape)
        x = x + rotary_pos_emb
        
        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                num_experts=self.num_experts,
                expert_dim=self.expert_dim,
                dtype=self.dtype)(x, deterministic)
        
        # Final layer norm
        x = nn.LayerNorm(dtype=self.dtype)(x)
        
        # Output projection
        return nn.Dense(self.vocab_size, dtype=self.dtype)(x)