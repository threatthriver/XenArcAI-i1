import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state # For loading state structure
import orbax.checkpoint as ocp
from transformers import AutoTokenizer
import os
import logging
import argparse

# Local imports
from config.config import get_config, AppConfig
from src.model import XenArcAI
from src.train import TrainState # Use the same TrainState definition

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_params(config: AppConfig):
    """Loads the model, tokenizer, and latest checkpoint."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Tokenizer pad token set to: {tokenizer.pad_token}")
    vocab_size = tokenizer.vocab_size # Get vocab size from tokenizer

    # Ensure model config vocab size matches tokenizer
    if config.model.vocab_size != vocab_size:
         logger.warning(f"Model config vocab_size ({config.model.vocab_size}) doesn't match tokenizer ({vocab_size}). Using tokenizer's size.")
         config = config.replace(model=config.model.replace(vocab_size=vocab_size))


    # Initialize model
    model = XenArcAI(config=config.model)
    key = jax.random.PRNGKey(0) # Dummy key for init

    # Create dummy state structure for Orbax restore
    # Optimizer state is not needed for inference, so we can use None or a dummy optax.GradientTransformation
    dummy_tx = optax.adam(1e-3) # Dummy optimizer
    dummy_input_ids = jnp.zeros((1, config.model.max_seq_len), dtype=jnp.int32)
    params = model.init(key, dummy_input_ids, deterministic=True)['params']
    state_structure = TrainState.create(apply_fn=model.apply, params=params, tx=dummy_tx)

    # Setup Orbax CheckpointManager
    mngr = ocp.CheckpointManager(
        os.path.abspath(config.training.model_dir),
        options=ocp.CheckpointManagerOptions(create=False) # Don't create if it doesn't exist
    )

    # Restore latest checkpoint
    latest_step = mngr.latest_step()
    params_restored = None
    if latest_step is not None:
        logger.info(f"Attempting to restore checkpoint from step {latest_step}...")
        restored_state = mngr.restore(latest_step, args=ocp.args.StandardRestore(state_structure))
        params_restored = restored_state.params
        logger.info(f"Restored parameters from step {latest_step}")
    else:
        logger.warning("No checkpoint found. Using initialized parameters for generation (likely untrained).")
        params_restored = params # Use initialized params

    mngr.close()
    return model, params_restored, tokenizer

@partial(jax.jit, static_argnums=(0,)) # Jit the function, static arg is model
def get_logits(model: nn.Module, params, input_ids: jnp.ndarray) -> jnp.ndarray:
    """Runs a forward pass to get logits."""
    # Add batch dimension if needed
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    logits = model.apply({'params': params}, input_ids, deterministic=True)
    # Return logits for the last token in the sequence
    return logits[:, -1, :] # Shape: (batch_size, vocab_size)

def generate_text(
    model: nn.Module,
    params,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    seed: int = 42
):
    """Generates text autoregressively."""
    key = jax.random.PRNGKey(seed)
    input_ids = tokenizer.encode(prompt, return_tensors='jax')[0] # Shape: (seq_len,)

    generated_ids = []

    logger.info(f"Generating text from prompt: '{prompt}'")
    for _ in range(max_new_tokens):
        # Get logits for the next token
        logits = get_logits(model, params, input_ids) # Shape: (1, vocab_size)

        # Apply temperature scaling
        scaled_logits = logits / temperature

        # Optional Top-K sampling
        if top_k is not None:
            top_k = min(top_k, scaled_logits.shape[-1])
            # Get the top_k logits and their indices
            top_logits, top_indices = jax.lax.top_k(scaled_logits, k=top_k)
            # Create a mask, setting logits outside top_k to -inf
            mask = jnp.full_like(scaled_logits, -jnp.inf).at[..., top_indices].set(top_logits)
            scaled_logits = mask

        # Sample next token ID using categorical distribution
        key, subkey = jax.random.split(key)
        next_token_id = jax.random.categorical(subkey, scaled_logits, axis=-1) # Shape: (1,)

        # Append generated token
        generated_ids.append(next_token_id.item())
        input_ids = jnp.concatenate([input_ids, next_token_id], axis=0)

        # Stop if EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            logger.info("EOS token generated. Stopping.")
            break

        # Optional: Trim input_ids if exceeding max_seq_len (simple sliding window)
        if input_ids.shape[0] > model.config.max_seq_len:
             input_ids = input_ids[-(model.config.max_seq_len):]


    # Decode generated IDs
    generated_text = tokenizer.decode(generated_ids)
    full_text = prompt + generated_text
    logger.info(f"Generated text:\n{full_text}")
    return full_text


def main():
    parser = argparse.ArgumentParser(description="Generate text using XenArcAI-v2 model.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, world!",
        help="The prompt to start generation from."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling. Higher values mean more randomness."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-K sampling. Only consider the top K most likely tokens."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation."
    )
    args = parser.parse_args()

    config = get_config()
    model, params, tokenizer = load_model_and_params(config)

    if params is None:
        logger.error("Failed to load parameters. Exiting.")
        return

    generate_text(
        model=model,
        params=params,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
