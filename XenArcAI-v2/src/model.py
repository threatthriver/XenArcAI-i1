import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Tuple
import functools

from ..config.config import ModelConfig # Use relative import

# --- Attention Mechanism ---
# NOTE: This is standard MultiHeadAttention. For true Flash Attention,
# specialized kernels (e.g., via Triton or custom CUDA) or libraries
# supporting it (like xformers if using PyTorch) are typically required.
# This implementation serves as a functional placeholder.
class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention (Placeholder for Flash Attention)."""
    config: ModelConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None, deterministic: bool = True):
        hidden_dim = self.config.num_heads * self.config.head_dim
        qkv_layer = nn.Dense(
            features=3 * hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name='qkv_projection'
        )
        qkv = qkv_layer(x)

        # Split Q, K, V
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # Reshape for multi-head attention
        batch_size, seq_len, _ = x.shape
        q = q.reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = nn.dot_product_attention_weights(
            q, k,
            bias=mask,
            dropout_rng=self.make_rng('dropout') if not deterministic else None,
            dropout_rate=self.config.dropout_rate,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None # Use default precision
        )

        # Apply attention weights to values
        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)

        # Reshape back to original dimensions
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)

        # Final linear projection
        output_layer = nn.Dense(
            features=hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name='output_projection'
        )
        output = output_layer(attn_output)
        return output

# --- Mixture of Experts ---
# NOTE: This is a simplified MoE layer with soft routing via softmax
# and basic MLP experts. A production MoE layer would typically involve:
# - Top-k gating/routing (selecting a subset of experts).
# - Load balancing mechanisms to ensure experts are utilized evenly.
# - Capacity factor to handle token routing limits.
class SimpleMoELayer(nn.Module):
    """Simplified Mixture of Experts (MoE) layer (Placeholder)."""
    config: ModelConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        hidden_dim = self.config.num_heads * self.config.head_dim
        batch_size, seq_len, _ = x.shape

        # Router (simple linear layer)
        router_logits = nn.Dense(
            features=self.config.num_experts,
            dtype=self.dtype,
            name='router'
        )(x) # Shape: (batch, seq_len, num_experts)
        routing_weights = jax.nn.softmax(router_logits, axis=-1) # Soft routing

        # Experts (simple MLPs)
        expert_outputs = []
        for i in range(self.config.num_experts):
            expert_mlp = nn.Sequential([
                nn.Dense(features=self.config.expert_dim, dtype=self.dtype, name=f'expert_{i}_dense1'),
                nn.relu,
                nn.Dense(features=hidden_dim, dtype=self.dtype, name=f'expert_{i}_dense2')
            ])
            expert_outputs.append(expert_mlp(x))

        # Stack expert outputs: (batch, seq_len, num_experts, hidden_dim)
        expert_outputs_stacked = jnp.stack(expert_outputs, axis=2)

        # Weighted sum of expert outputs
        # routing_weights shape: (batch, seq_len, num_experts)
        # expert_outputs_stacked shape: (batch, seq_len, num_experts, hidden_dim)
        # Need to expand routing_weights for broadcasting
        weighted_output = jnp.einsum(
            'bse,bseh->bsh',
            routing_weights,
            expert_outputs_stacked
        )

        return weighted_output

# --- Transformer Block ---
class TransformerBlock(nn.Module):
    """A single block of the Transformer model."""
    config: ModelConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None, deterministic: bool = True):
        hidden_dim = self.config.num_heads * self.config.head_dim

        # Attention sub-layer
        attn_input = nn.LayerNorm(dtype=self.dtype, name='ln_1')(x)
        attn_output = MultiHeadAttention(
            config=self.config,
            dtype=self.dtype,
            name='attention'
        )(attn_input, mask=mask, deterministic=deterministic)
        attn_output = nn.Dropout(
            rate=self.config.dropout_rate,
            deterministic=deterministic
        )(attn_output)
        x = x + attn_output # Residual connection

        # MoE/FFN sub-layer
        ffn_input = nn.LayerNorm(dtype=self.dtype, name='ln_2')(x)
        # --- Use MoE ---
        ffn_output = SimpleMoELayer(
             config=self.config,
             dtype=self.dtype,
             name='moe_layer'
        )(ffn_input)
        # --- OR Use Standard FFN ---
        # ffn_output = nn.Sequential([
        #     nn.Dense(features=self.config.mlp_dim, dtype=self.dtype, name='ffn_dense1'),
        #     nn.relu,
        #     nn.Dropout(rate=self.config.dropout_rate, deterministic=deterministic),
        #     nn.Dense(features=hidden_dim, dtype=self.dtype, name='ffn_dense2'),
        # ], name='ffn')(ffn_input)

        ffn_output = nn.Dropout(
            rate=self.config.dropout_rate,
            deterministic=deterministic
        )(ffn_output)
        x = x + ffn_output # Residual connection

        return x

# --- Rotary Position Embedding ---
def rotary_pos_emb(x, seq_dim=1, max_wavelength=10000.0):
    """
    Applies Rotary Position Embedding (RoPE) to the input tensor.

    RoPE encodes absolute positional information using rotation matrices
    applied in the embedding space. It has shown effectiveness in various
    Transformer models.

    Args:
        x: Input tensor, shape (..., seq_len, dim).
        seq_dim: The axis corresponding to the sequence length.
        max_wavelength: The maximum wavelength for the sinusoidal functions.
                        Controls the frequency range.

    Returns:
        Tensor with RoPE applied.
    """
    seq_len = x.shape[seq_dim]
    hidden_dim = x.shape[-1]
    assert hidden_dim % 2 == 0, "Hidden dimension must be even for Rotary Embedding"

    # Generate frequency bands (thetas)
    freq_exponents = jnp.arange(0, hidden_dim, 2, dtype=jnp.float32) / hidden_dim
    inv_freq = 1.0 / (max_wavelength ** freq_exponents)

    # Generate position indices
    positions = jnp.arange(seq_len, dtype=jnp.float32)

    # Calculate angles (theta * position)
    # positions shape: (seq_len,)
    # inv_freq shape: (hidden_dim/2,)
    # freqs shape: (seq_len, hidden_dim/2)
    freqs = jnp.einsum('i,j->ij', positions, inv_freq)

    # Create cosine and sine embeddings
    # emb shape: (seq_len, hidden_dim)
    emb = jnp.concatenate((freqs, freqs), axis=-1)

    # Reshape for broadcasting
    while emb.ndim < x.ndim:
        emb = emb[jnp.newaxis, ...]
    # Swap axes if seq_dim is not 1
    if seq_dim != 1:
         emb = jnp.swapaxes(emb, 1, seq_dim)

    # Split into real and imaginary parts (cos and sin)
    cos_emb = jnp.cos(emb)
    sin_emb = jnp.sin(emb)

    # Apply rotation to the input tensor x
    # x has shape (..., seq_len, hidden_dim)
    # Split x into two halves for rotation: x1, x2 shape (..., seq_len, hidden_dim/2)
    x1, x2 = jnp.split(x, 2, axis=-1)

    # Create the rotated version of x: (-x2, x1)
    # rotated_x shape: (..., seq_len, hidden_dim)
    rotated_x = jnp.concatenate((-x2, x1), axis=-1)

    # Combine using cosine and sine embeddings
    # result = x * cos(theta*pos) + rotated_x * sin(theta*pos)
    return x * cos_emb + rotated_x * sin_emb


# --- Main Model ---
class XenArcAI(nn.Module):
    """XenArcAI Language Model."""
    config: ModelConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, deterministic: bool = True):
        dtype = jnp.dtype(self.config.dtype)
        hidden_dim = self.config.num_heads * self.config.head_dim

        # 1. Input Embedding
        embed_layer = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=hidden_dim,
            dtype=dtype,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name='token_embedder'
        )
        x = embed_layer(input_ids.astype('int32'))

        # 2. Apply Rotary Position Embeddings
        x = rotary_pos_emb(x, seq_dim=1) # Apply RoPE

        # 3. Dropout after embeddings
        x = nn.Dropout(
            rate=self.config.dropout_rate,
            deterministic=deterministic
        )(x)

        # 4. Transformer Blocks
        # Create attention mask if not provided (for causal LM)
        if attention_mask is None:
             attention_mask = nn.make_causal_mask(input_ids)
        # Add batch dimension if needed
        if attention_mask.ndim == 2:
            attention_mask = attention_mask[jnp.newaxis, ...]

        for i in range(self.config.num_layers):
            x = TransformerBlock(
                config=self.config,
                dtype=dtype,
                name=f'transformer_block_{i}'
            )(x, mask=attention_mask, deterministic=deterministic)

        # 5. Final Layer Normalization
        x = nn.LayerNorm(dtype=dtype, name='final_ln')(x)

        # 6. Output Projection (Logits)
        logits = nn.Dense(
            features=self.config.vocab_size,
            dtype=dtype, # Output logits often kept in float32
            kernel_init=nn.initializers.xavier_uniform(),
            name='output_projection'
        )(x)

        return logits
