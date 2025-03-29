import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, checkpoints # Use checkpoints directly
import orbax.checkpoint as ocp # Orbax for checkpointing
import optax
from tqdm import tqdm
import os
import logging
from functools import partial
import time

# Local imports
from ..config.config import get_config, AppConfig # Use relative import
from .model import XenArcAI # Use relative import
from .data_pipeline import create_data_pipeline, get_dataset_iterator # Use relative import

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Train State ---
class TrainState(train_state.TrainState):
    # You can add more things to track here if needed, e.g., dropout PRNG key
    # Add other state variables if needed, e.g., dropout PRNG key
    dropout_rng: Optional[jax.random.PRNGKey] = None

# --- Loss Function ---
def compute_loss(logits, labels, vocab_size):
    """Calculates cross-entropy loss for language modeling."""
    one_hot_labels = jax.nn.one_hot(labels, num_classes=vocab_size)
    # Ensure logits and labels have compatible shapes
    # Logits: (batch, seq_len, vocab_size)
    # Labels: (batch, seq_len) -> One-hot: (batch, seq_len, vocab_size)
    loss = -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    # Average loss over the sequence length (axis=1) and batch (axis=0)
    # Mask out padding tokens if necessary (using attention_mask or labels != pad_token_id)
    # For simplicity here, we assume labels are not padded or padding doesn't affect loss significantly
    return jnp.mean(loss)

# --- Training Step ---
@partial(jax.pmap, axis_name='batch') # Decorator for pmap
def train_step(state: TrainState, batch: Dict[str, jnp.ndarray], model: nn.Module, config: AppConfig):
    """Performs a single training step, designed to be pmapped."""
    # Assumes batch contains 'input_ids'. For Causal LM, labels are shifted inputs.
    # However, standard practice is often to use input_ids directly as labels
    # and let the model's causal mask handle the shifting implicitly during attention.
    labels = batch['input_ids']

    # Generate new dropout key for this step
    dropout_rng = jax.random.fold_in(state.dropout_rng, state.step)

    def loss_fn(params):
        logits = model.apply(
            {'params': params},
            batch['input_ids'],
            deterministic=False, # Enable dropout during training
            rngs={'dropout': dropout_rng}
        )
        loss = compute_loss(logits, labels, config.model.vocab_size)
        return loss

    # Calculate gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # === Gradient Synchronization (Implicit in pmap) ===
    # jax.lax.pmean automatically averages gradients across devices
    # when used within a pmapped function with axis_name='batch'.
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch') # Average loss across devices too

    # Update state (optimizer step)
    new_state = state.apply_gradients(grads=grads)

    # Update dropout RNG for the next step (required if stateful)
    # new_state = new_state.replace(dropout_rng=dropout_key) # If storing RNG in state

    metrics = {'loss': loss}
    return new_state, metrics

# --- Evaluation Step ---
@partial(jax.pmap, axis_name='batch') # Decorator for pmap
def eval_step(state: TrainState, batch: Dict[str, jnp.ndarray], model: nn.Module, config: AppConfig):
    """Performs a single evaluation step, designed to be pmapped."""
    labels = batch['input_ids']
    logits = model.apply(
        {'params': state.params},
        batch['input_ids'],
        deterministic=True # Disable dropout during evaluation
    )
    loss = compute_loss(logits, labels, config.model.vocab_size)

    # Average loss across the devices participating in this pmap call
    metrics = {'loss': jax.lax.pmean(loss, axis_name='batch')}
    return metrics


# --- Main Training Function ---
def train():
    config = get_config()
    num_devices = jax.local_device_count()
    logger.info(f"Starting training on {num_devices} devices.")

    # --- Setup ---
    key = jax.random.PRNGKey(config.training.seed)
    model_key, params_key, dropout_key = jax.random.split(key, 3)

    # Create directories
    os.makedirs(config.training.model_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)

    # --- Model Initialization ---
    model = XenArcAI(config=config.model)
    dummy_input_ids = jnp.zeros(
        (config.data.batch_size_per_device, config.data.max_seq_length),
        dtype=jnp.int32
    )
    params = model.init(params_key, dummy_input_ids, deterministic=True)['params']
    logger.info("Model initialized.")
    # Log parameter count
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"Model parameter count: {param_count / 1e6:.2f}M")

    # --- Optimizer ---
    # AdamW optimizer with a learning rate schedule (warmup + cosine decay)
    # Calculate total expected training steps (approximate)
    # This needs adjustment based on the actual dataset size and number of epochs.
    # Using a placeholder value if dataset size is unknown.
    steps_per_epoch_approx = 10000 // (config.data.batch_size_per_device * num_devices) # Example placeholder
    total_train_steps = config.training.num_train_epochs * steps_per_epoch_approx
    logger.info(f"Approximate total training steps: {total_train_steps}")
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        decay_steps=max(1, total_train_steps - config.training.warmup_steps), # Ensure decay_steps >= 1
        end_value=config.training.learning_rate * 0.01, # Decay to 1% of peak LR
    )
    optimizer = optax.adamw(learning_rate=schedule)

    # --- Train State ---
    # Pass the dropout PRNG key to the state if needed for stateful dropout
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        dropout_rng=dropout_key # Store base dropout key in state
    )
    logger.info("Train state created.")

    # --- Checkpointing (using Orbax) ---
    # Abstract Checkpoint Manager for potentially async saving
    mngr = ocp.CheckpointManager(
        os.path.abspath(config.training.model_dir),
        options=ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=config.training.checkpoint_steps)
    )
    # Restore the latest checkpoint if available
    latest_step = mngr.latest_step()
    if latest_step is not None:
        logger.info(f"Attempting to restore checkpoint from step {latest_step}...")
        # Target needs to be structured like the saved object (TrainState without apply_fn)
        target_state_for_restore = TrainState.create(apply_fn=None, params=params, tx=optimizer, dropout_rng=dropout_key)
        restored_state = mngr.restore(latest_step, args=ocp.args.StandardRestore(target_state_for_restore))
        state = restored_state # Overwrite initial state if restore successful
        logger.info(f"Restored checkpoint from step {latest_step}")
        # Ensure dropout RNG is updated if restored state doesn't include it or needs splitting
        state = state.replace(dropout_rng=dropout_key)
    else:
        logger.info("No checkpoint found, starting from scratch.")


    # --- Data Loading ---
    logger.info("Setting up data pipelines...")
    train_ds = create_data_pipeline(config.data, split='train')
    eval_ds = create_data_pipeline(config.data, split='validation')
    train_iter = get_dataset_iterator(train_ds)
    eval_iter = get_dataset_iterator(eval_ds)
    logger.info("Data pipelines ready.")

    # --- Parallelize Training & Eval Steps ---
    # The train_step and eval_step functions are already decorated with pmap
    p_train_step = train_step
    p_eval_step = eval_step

    # Replicate state across devices AFTER potential restore
    state = jax.device_put_replicated(state, jax.local_devices())
    logger.info("State replicated across devices.")

    # --- Training Loop ---
    logger.info("Starting training loop...")
    start_step = int(state.step[0]) # Get step from replicated state (device 0)
    global_step = start_step
    train_start_time = time.time()

    for epoch in range(config.training.num_train_epochs):
        logger.info(f"--- Epoch {epoch+1}/{config.training.num_train_epochs} ---")
        train_metrics_accum = []
        epoch_start_time = time.time()

        # Use tqdm for progress bar over batches in an epoch
        # Note: If using iterable dataset, total steps might be unknown.
        with tqdm(initial=global_step, total=total_train_steps, desc=f"Epoch {epoch+1}", unit="step") as pbar:
            try:
                while True: # Loop through batches from the iterator
                    batch_start_time = time.time()
                    batch = next(train_iter) # Get the next batch

                    # Shard batch across devices (manually)
                    # Assumes batch is a dictionary of arrays
                    sharded_batch = jax.tree_map(
                        lambda x: x.reshape((num_devices, -1) + x.shape[1:]), batch
                    )

                    # Perform a training step (already pmapped)
                    state, metrics = p_train_step(state, sharded_batch, model, config)
                    train_metrics_accum.append(metrics)
                    current_step = int(state.step[0]) # Get current step from replicated state
                    pbar.update(current_step - global_step) # Update progress bar by steps taken
                    global_step = current_step

                    batch_time = time.time() - batch_start_time

                    # Logging
                    if global_step % config.training.log_steps == 0 and train_metrics_accum:
                        # Aggregate metrics (take mean across steps and devices)
                        # Metrics are already averaged across devices in p_train_step
                        avg_loss = jnp.mean(jnp.array([m['loss'] for m in train_metrics_accum])).item()
                        steps_per_sec = config.training.log_steps / (time.time() - train_start_time)
                        pbar.set_postfix(loss=f"{avg_loss:.4f}", sps=f"{steps_per_sec:.2f}")
                        logger.info(f"Step: {global_step}, Train Loss: {avg_loss:.4f}, Steps/sec: {steps_per_sec:.2f}")
                        train_metrics_accum = [] # Reset accumulator
                        train_start_time = time.time() # Reset timer

                    # Checkpointing (delegated to CheckpointManager)
                    # Save unreplicated state from device 0
                    unreplicated_state = jax.device_get(jax.tree_map(lambda x: x[0], state))
                    mngr.save(global_step, args=ocp.args.StandardSave(unreplicated_state))


                    # Evaluation
                    if global_step % config.training.eval_steps == 0:
                        logger.info(f"Running evaluation at step {global_step}...")
                        eval_losses = []
                        eval_start_time = time.time()
                        eval_iter = get_dataset_iterator(eval_ds) # Re-get iterator
                        try:
                            while True: # Evaluate on the entire eval dataset
                                eval_batch = next(eval_iter)
                                sharded_eval_batch = jax.tree_map(
                                    lambda x: x.reshape((num_devices, -1) + x.shape[1:]), eval_batch
                                )
                                metrics = p_eval_step(state, sharded_eval_batch, model, config)
                                # metrics['loss'] is already averaged across devices by pmap
                                eval_losses.append(metrics['loss'])
                        except StopIteration:
                            pass # End of eval dataset

                        if eval_losses:
                            # Average the loss across all eval batches
                            avg_eval_loss = jnp.mean(jnp.array(eval_losses)).item()
                            eval_time = time.time() - eval_start_time
                            logger.info(f"Step: {global_step}, Eval Loss: {avg_eval_loss:.4f}, Eval Time: {eval_time:.2f}s")
                        else:
                            logger.warning(f"Step: {global_step}, No batches found in eval dataset iterator.")


            except StopIteration:
                logger.info(f"End of training data for epoch {epoch+1}.")
                # Re-initialize train iterator for the next epoch if needed
                train_iter = get_dataset_iterator(train_ds)

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} finished in {epoch_time:.2f}s")
        # Wait for checkpoint saving to complete before starting next epoch
        mngr.wait_until_finished()


    logger.info("Training finished!")
    # Final checkpoint save (optional, as manager saves periodically)
    logger.info("Saving final checkpoint...")
    unreplicated_state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    mngr.save(global_step, args=ocp.args.StandardSave(unreplicated_state), force=True)
    mngr.wait_until_finished() # Ensure final save completes
    mngr.close() # Close the manager
    logger.info(f"Final checkpoint saved at step {global_step}.")


if __name__ == "__main__":
    train()
