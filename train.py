import argparse
import logging
import math
import os
import time
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

# XLA specific imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.debug.metrics as met
import torch_xla.runtime as xr
xr.use_spmd() # Enable SPMD execution mode for potential performance benefits

# Hugging Face imports
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW

# Model import (assuming model.py is in the same directory or PYTHON PATH)
from model import XenArcAIi1, ModelArgs # Assuming XenArcAIi1 initializes its own tokenizer

# Monitoring
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    wandb = None # Define wandb as None if not available

# Other necessary imports
import numpy as np
from safetensors.torch import save_file, load_file # Model uses safetensors directly


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---

@dataclass
class TrainingArgs:
    # --- Model & Tokenizer ---
    # ModelArgs are now nested, defaults taken from model.py unless overridden
    model_config: ModelArgs = field(default_factory=ModelArgs)
    # Tokenizer is loaded FROM the instantiated model, no separate path needed here.
    # `model.py` should handle finding/loading/creating the tokenizer.
    resume_from_checkpoint: Optional[str] = None # Path to checkpoint dir to resume (e.g., checkpoints/xenarc_trained/step_1000)
    save_dir: str = "checkpoints/xenarc_trained"

    # --- Dataset ---
    # REPLACE with your actual HF dataset identifier
    dataset_name: str = "your_hf_username/your_preprocessed_hinglish_dataset"
    dataset_streaming: bool = True # Use streaming for large datasets
    train_split: str = "train" # Or "train_streaming" etc.
    validation_split: Optional[str] = "validation" # Optional validation split
    text_column: str = "text"
    # Columns needed for 70/30 router label handling
    router_label_column: str = "router_labels" # List/Tensor of expert indices per token
    has_router_label_column: str = "has_router_labels" # Boolean indicating if router_labels are valid
    language_column: str = "language" # e.g., "english", "hindi", "hinglish"
    shuffle_buffer_size: int = 10000 # Buffer size for streaming dataset shuffling

    # --- Training Hyperparameters ---
    learning_rate: float = 5e-5 # Common starting point, adjust based on experiments
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98 # Often 0.98 for AdamW variants
    adam_epsilon: float = 1e-6
    max_grad_norm: float = 1.0 # Clipping handled by xm.optimizer_step
    num_train_epochs: int = 1 # Adjust as needed, often less for huge datasets
    max_steps: Optional[int] = None # If set, overrides num_train_epochs
    lr_scheduler_type: str = "linear" # "linear", "cosine", etc.
    warmup_ratio: float = 0.03 # Percentage of total steps for warmup

    # --- Batching & Data Loading ---
    per_device_train_batch_size: int = 8  # Adjust based on TPU core memory (e.g., v3-8: 4-8, v4: 8-16+)
    per_device_eval_batch_size: int = 8   # Batch size for evaluation
    gradient_accumulation_steps: int = 16 # Effective BS = N_DEVICES * BATCH_SIZE * ACCUM_STEPS
    dataloader_num_workers: int = 4 # Depends on TPU host CPU/memory, 4-8 is common
    # Sequence length during training - MUST be compatible with model's RoPE/attention if fixed size
    # ModelArgs.max_seq_len might be theoretical max, choose a practical training length
    # Ensure this is <= ModelArgs.initial_seq_len if not using dynamic RoPE scaling during training
    train_max_seq_length: int = 4096

    # --- Logging & Saving ---
    logging_steps: int = 20  # Log metrics every N global steps
    save_steps: int = 1000 # Save checkpoint every N global steps
    eval_steps: int = 1000 # Evaluate every N global steps (if validation_split is provided)

    # --- W&B Monitoring ---
    use_wandb: bool = True # Enable/disable WandB
    wandb_project: str = "XenArcAI_TPU_Training_SPMD"
    wandb_run_name: Optional[str] = None # Auto-generated if None
    wandb_entity: Optional[str] = None # Your WandB username or team name

    # --- TPU Settings ---
    seed: int = 42
    bf16: bool = True # Use bfloat16 precision (native on TPUs)


# Global tokenizer variable needed for dataset mapping function
# Will be initialized in _mp_fn for each process
# Note: Sharing complex Python objects across processes can be tricky.
# It's often safer to initialize within each process.
# We will pass the tokenizer to the map function instead.

def set_seed(seed: int):
    """Sets random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # No torch.cuda specifics needed for XLA

def save_model_checkpoint(model: XenArcAIi1, tokenizer, save_path: str, args: TrainingArgs):
    """Saves model and tokenizer state using model's save method."""
    os.makedirs(save_path, exist_ok=True)
    model_save_path = os.path.join(save_path, "xenarc_model") # Base path for model parts
    model.save(model_save_path) # Model handles safetensor splitting
    # Save tokenizer separately
    tokenizer.save_pretrained(save_path)
    # Save training args as well for reproducibility
    torch.save(args, os.path.join(save_path, "training_args.bin"))
    logger.info(f"Model, tokenizer, and args saved to {save_path}")

# --- Data Preprocessing ---

def preprocess_function(examples: Dict[str, Any], tokenizer: AutoTokenizer, args: TrainingArgs) -> Dict[str, Any]:
    """
    Tokenizes text, prepares router labels according to has_router_labels flag,
    and handles padding/truncation for a single example.
    """
    max_len = args.train_max_seq_length

    # Tokenize text
    # Use padding='max_length' and truncation=True
    tokenized_inputs = tokenizer(
        examples[args.text_column],
        max_length=max_len,
        padding="max_length", # Pad to max_len
        truncation=True,
        return_tensors=None # Don't return tensors yet, handle list conversion below
    )

    # Language identifier
    language = examples.get(args.language_column, "unknown")

    # --- Router Label Processing (70/30 logic) ---
    has_labels = examples.get(args.has_router_label_column, False) is True
    router_labels_raw = examples.get(args.router_label_column, [])

    # Initialize padded labels with ignore index (-100)
    padded_router_labels = [-100] * max_len

    if has_labels and router_labels_raw:
        # Truncate labels if longer than max_len
        label_len = min(len(router_labels_raw), max_len)
        # Assign valid labels, keep the rest as -100
        padded_router_labels[:label_len] = router_labels_raw[:label_len]
    # Else: Keep all labels as -100 (unlabeled example or missing labels)

    processed = {
        # Return as lists first, convert to tensors in collate_fn or later if needed
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "router_labels": padded_router_labels,
        "language": language, # Keep language as string
        "has_router_labels": has_labels # Pass this info if needed downstream
    }
    return processed


# --- Main Training Function (per TPU core) ---

def _mp_fn(index: int, args: TrainingArgs):
    """Main function executed on each TPU core."""
    # Set device and seed correctly for distributed training
    device = xm.xla_device()
    set_seed(args.seed + index) # Ensure different seed per process
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    is_master = xm.is_master_ordinal()

    # Barrier to ensure all processes start together
    xm.rendezvous('init')

    if is_master:
        logger.info("--- XenArcAIi1 TPU Training Start ---")
        logger.info(f"Running on {world_size} TPU cores.")
        logger.info(f"Training Arguments: {args}")
        # Create save directory if it doesn't exist
        os.makedirs(args.save_dir, exist_ok=True)

    # === 1. Initialize Model ===
    # Model initialization includes tokenizer loading/creation as per model.py logic
    xm.master_print("Initializing Model and Tokenizer...")
    try:
        # Pass the nested ModelArgs dataclass
        model = XenArcAIi1(args.model_config)
        tokenizer = model.tokenizer # Get tokenizer from the model instance
        xm.master_print(f"Model and Tokenizer initialized. Vocab size: {len(tokenizer)}")
    except Exception as e:
        logger.error(f"Failed to initialize model or tokenizer: {e}", exc_info=True)
        raise # Stop execution if model fails

    # Move model to XLA device
    model.to(device)

    # === 2. Load Dataset ===
    xm.master_print(f"Loading dataset '{args.dataset_name}'...")
    # Load the dataset within each process for IterableDataset compatibility
    try:
        raw_dataset = load_dataset(
            args.dataset_name,
            split=args.train_split,
            streaming=args.dataset_streaming
        )
        if not isinstance(raw_dataset, IterableDataset):
             raise TypeError("Dataset must be loaded in streaming mode (`IterableDataset`) for this script.")
        xm.master_print(f"Dataset '{args.dataset_name}' loaded successfully (streaming).")
    except Exception as e:
        logger.error(f"Failed to load dataset '{args.dataset_name}': {e}", exc_info=True)
        raise

    # Shuffle the stream if needed
    # Note: Shuffling happens BEFORE sharding in typical setups.
    # If using xr.DataParallel() later, it might handle sharding implicitly.
    # Shuffling large streams requires a buffer.
    train_dataset = raw_dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
    xm.master_print(f"Dataset shuffled with buffer size {args.shuffle_buffer_size}.")

    # Define the preprocessing function with the correct tokenizer for this process
    # Use partial or lambda to pass tokenizer and args
    from functools import partial
    bound_preprocess_function = partial(preprocess_function, tokenizer=tokenizer, args=args)

    # Apply preprocessing
    # Use batched=False for potentially simpler handling per example, adjust if needed
    train_dataset = train_dataset.map(bound_preprocess_function, batched=False)
    xm.master_print("Dataset preprocessing applied.")

    # Basic check on the first element (only on master to avoid redundancy)
    if is_master:
        try:
            first_element = next(iter(train_dataset))
            xm.master_print(f"First preprocessed element keys: {list(first_element.keys())}")
            xm.master_print(f"Sample input_ids length: {len(first_element['input_ids'])}")
            xm.master_print(f"Sample router_labels length: {len(first_element['router_labels'])}")
            assert len(first_element['input_ids']) == args.train_max_seq_length
            assert len(first_element['router_labels']) == args.train_max_seq_length
        except Exception as e:
            logger.error(f"Error inspecting first dataset element: {e}", exc_info=True)
            # Potentially raise here if the dataset seems invalid

    # Remove columns not needed by the model (optional, might save memory)
    # train_dataset = train_dataset.remove_columns([...])

    # === 3. Create DataLoader ===
    # Needs careful handling for distributed IterableDataset
    # torch_xla provides helpers or manual sharding might be needed
    # Let's try standard DataLoader first, DataParallel might be needed if issues arise
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
        # Collate function might be needed if returning lists from preprocess
        collate_fn=None # Use default collate if preprocess returns tensors directly (needs adjustment above)
                        # Or define a custom collate_fn to convert lists to tensors here
    )
    # Wrap DataLoader for distributed execution (SPMD requires sharding)
    # This automatically shards the data across devices
    # train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
    # Update: Using xr.DataParallel for SPMD
    train_loader = xr.DataParallel(train_loader, device_ids=list(range(world_size)))


    # (Optional) Validation DataLoader setup (similar process)
    eval_loader = None
    if args.validation_split:
        try:
            eval_dataset_raw = load_dataset(args.dataset_name, split=args.validation_split, streaming=args.dataset_streaming)
            if isinstance(eval_dataset_raw, IterableDataset):
                 # No need to shuffle validation data
                 eval_dataset = eval_dataset_raw.map(bound_preprocess_function, batched=False)
                 eval_loader = DataLoader(
                     eval_dataset,
                     batch_size=args.per_device_eval_batch_size,
                     num_workers=args.dataloader_num_workers,
                     collate_fn=None # Adjust if needed
                 )
                 eval_loader = xr.DataParallel(eval_loader, device_ids=list(range(world_size)))
                 xm.master_print(f"Validation dataset '{args.validation_split}' loaded and prepared.")
            else:
                 xm.master_print(f"Validation split '{args.validation_split}' is not an IterableDataset, skipping validation.")

        except Exception as e:
            logger.warning(f"Could not load or prepare validation split '{args.validation_split}': {e}. Skipping validation.")


    # === 4. Initialize Optimizer and Scheduler ===
    # Filter parameters that require gradients
    no_decay = ["bias", "LayerNorm.weight", "RMSNorm.weight"] # Parameters exempt from weight decay
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )
    xm.master_print("Optimizer AdamW initialized.")

    # Calculate total training steps
    # Note: With IterableDataset, length is often unknown. Estimate or use max_steps.
    if args.max_steps:
        num_training_steps = args.max_steps
        num_epochs = math.ceil(args.max_steps / (estimated_steps_per_epoch if 'estimated_steps_per_epoch' in locals() else 10000)) # Rough estimate
        xm.master_print(f"Training for a maximum of {args.max_steps} steps.")
    else:
        # Estimate steps if possible (very hard with streaming)
        # Let's default to a large number if max_steps isn't set, or require max_steps with streaming
        logger.warning("Length of streaming dataset is unknown. `max_steps` is recommended for scheduler calculation. Estimating steps.")
        # Use a placeholder large number - refine if you have a better estimate
        estimated_total_samples = 1_000_000_000 # Example: 1 Billion samples estimate
        steps_per_epoch_est = estimated_total_samples // (args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps)
        num_training_steps = int(steps_per_epoch_est * args.num_train_epochs)
        args.max_steps = num_training_steps # Update args.max_steps based on estimate
        xm.master_print(f"Estimated total training steps: {num_training_steps} (for {args.num_train_epochs} epochs)")

    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    xm.master_print(f"LR Scheduler initialized with {num_warmup_steps} warmup steps.")

    # === 5. Resume from Checkpoint (if applicable) ===
    # Must load optimizer/scheduler states AFTER they are initialized
    start_step = 0
    if args.resume_from_checkpoint:
        trainer_state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.pt")
        model_load_path = os.path.join(args.resume_from_checkpoint, "xenarc_model") # Base path for model parts

        if os.path.exists(trainer_state_path) and os.path.isdir(model_load_path): # Check model dir exists too
            xm.master_print(f"Attempting to load checkpoint from: {args.resume_from_checkpoint}")

            # Load Model Weights (Master Only - then broadcast or ensure consistency)
            # Since model loading might be complex and involves custom logic,
            # let's perform it BEFORE distributing the model state, if possible, or ensure
            # the loading logic itself is robust to distributed settings.
            # Model's load method needs to be called carefully. Assuming it loads correctly:
            if is_master:
                try:
                    # Re-instantiate model from saved config might be safer
                    saved_args_path = os.path.join(args.resume_from_checkpoint, "training_args.bin")
                    if os.path.exists(saved_args_path):
                        loaded_train_args = torch.load(saved_args_path)
                        # Ensure loaded model config is compatible or use it directly
                        model = XenArcAIi1.load(model_load_path, loaded_train_args.model_config)
                        tokenizer = model.tokenizer # Update tokenizer ref
                        xm.master_print("Loaded model weights and tokenizer from checkpoint.")
                    else:
                         xm.master_print("WARNING: training_args.bin not found in checkpoint. Loading model with current args.", logging.WARNING)
                         model = XenArcAIi1.load(model_load_path, args.model_config)
                         tokenizer = model.tokenizer

                    # Re-initialize preprocess function with potentially new tokenizer
                    bound_preprocess_function = partial(preprocess_function, tokenizer=tokenizer, args=args)
                    # Re-create datasets/loaders if tokenizer changed significantly? For safety, assume it's compatible.

                except Exception as e:
                    logger.error(f"Failed to load model from checkpoint path {model_load_path}: {e}. Starting fresh.", exc_info=True)
                    # Re-initialize model if loading failed
                    model = XenArcAIi1(args.model_config)
                    tokenizer = model.tokenizer


            # Barrier to ensure master finishes loading model before others proceed/load trainer state
            xm.rendezvous("model_load")
            # Broadcast model state from master IF necessary (XLA often handles this implicitly)
            # xm.broadcast_master_param(model) # Consider if needed

            model.to(device) # Ensure model is on device after potential reload

            # Load Optimizer/Scheduler State (All Processes)
            try:
                trainer_state = torch.load(trainer_state_path, map_location="cpu")
                optimizer.load_state_dict(trainer_state['optimizer'])
                scheduler.load_state_dict(trainer_state['scheduler'])
                start_step = trainer_state['step'] + 1 # Resume from next step
                xm.master_print(f"Loaded optimizer and scheduler state. Resuming from global step {start_step}")
            except Exception as e:
                logger.error(f"Failed to load optimizer/scheduler state from {trainer_state_path}: {e}. Starting from step 0.", exc_info=True)
                start_step = 0
        else:
            xm.master_print("Checkpoint specified but not found or incomplete. Starting training from scratch.")

    # Barrier after checkpoint loading
    xm.rendezvous("load_checkpoint_done")

    # === 6. Initialize WandB (Master Only) ===
    if is_master and args.use_wandb:
        if _WANDB_AVAILABLE:
            try:
                run_name = args.wandb_run_name or f"xenarc_tpu_{time.strftime('%Y%m%d_%H%M%S')}"
                wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    entity=args.wandb_entity, # Your WandB username or team
                    config=vars(args) # Log training arguments
                )
                logger.info(f"WandB initialized. Project: {args.wandb_project}, Run: {run_name}")
            except Exception as e:
                logger.error(f"Failed to initialize WandB: {e}", exc_info=True)
                args.use_wandb = False # Disable wandb if init fails
        else:
            logger.warning("WandB is enabled in args, but the library is not installed. Skipping WandB.")
            args.use_wandb = False

    # === 7. Training Loop ===
    xm.master_print("Starting training loop...")
    global_step = start_step
    train_loss = torch.tensor(0.0, device=device)
    # Trackers for averaging loss/rewards over logging steps
    accumulated_loss = 0.0
    accumulated_lm_loss = 0.0
    accumulated_balance_loss = 0.0
    accumulated_supervised_loss = 0.0
    accumulated_router_reward = 0.0
    accumulated_attn_reward = 0.0


    # Loop indefinitely, break based on max_steps
    # Use train_loader which is sharded via xr.DataParallel
    # The loader yields batches specific to each device
    for step, batch in enumerate(train_loader): # xr.DataParallel handles the device-specific batching

        # Resume logic: Skip steps already completed
        if global_step < start_step:
             if (global_step + 1) % args.gradient_accumulation_steps == 0:
                 # Need to step scheduler even when skipping to keep it sync
                 # scheduler.step() # This might desync if LR changed significantly, be careful
                 pass # Avoid stepping scheduler during skip phase? Check best practice.
             global_step += 1
             continue

        model.train() # Set model to training mode

        # Prepare batch data (move to device, ensure correct types)
        # Assuming collate_fn handles tensor conversion or batch is already tensors
        try:
            input_ids = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            # Router labels may have ignore index -100
            router_labels = batch['router_labels'].to(device, dtype=torch.long)
            # Language info - model expects list of strings
            # Batching strings needs custom collate or handle per-device list
            # Assuming 'language' is available per device if needed by model logic
            languages = batch.get('language', None) # xr.DataParallel might replicate strings

        except Exception as e:
             logger.error(f"[Rank {rank}] Error processing batch at step {global_step}: {e}", exc_info=True)
             # Skip corrupted batch? Or raise?
             continue # Skip this batch

        # Forward pass
        with torch.autocast(device_type='xla', dtype=torch.bfloat16 if args.bf16 else torch.float):
            # Pass router_labels with potential -100 ignore index
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                router_labels=router_labels, # Pass directly, model should handle ignore index
                languages=languages # Pass language info if model uses it
            )
            logits, total_usage, total_balance_loss, total_supervised_loss, total_router_reward, total_attn_reward = outputs

            # Calculate Language Modeling Loss (Cross-Entropy)
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            # Flatten the tokens
            vocab_size = shift_logits.size(-1)
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id # Ignore padding tokens in LM loss
            )

            # Combine losses (apply weights if needed, though already weighted in args?)
            # Ensure losses are valid tensors before combining
            total_balance_loss = total_balance_loss if torch.is_tensor(total_balance_loss) else torch.tensor(0.0, device=device)
            total_supervised_loss = total_supervised_loss if torch.is_tensor(total_supervised_loss) else torch.tensor(0.0, device=device)

            loss = lm_loss + total_balance_loss + total_supervised_loss

        # Accumulate metrics
        accumulated_loss += loss.item() # Use .item() to get float value
        accumulated_lm_loss += lm_loss.item()
        accumulated_balance_loss += total_balance_loss.item()
        accumulated_supervised_loss += total_supervised_loss.item()
        accumulated_router_reward += total_router_reward.item() if torch.is_tensor(total_router_reward) else 0.0
        accumulated_attn_reward += total_attn_reward.item() if torch.is_tensor(total_attn_reward) else 0.0

        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        loss.backward() # Compute gradients

        # --- Gradient Accumulation Step ---
        if (global_step + 1) % args.gradient_accumulation_steps == 0:
            # Optimizer Step (handles clipping and synchronization)
            optimizer_step_start_time = time.time()
            xm.optimizer_step(optimizer, barrier=True) # Barrier ensures sync
            optimizer_step_duration = time.time() - optimizer_step_start_time

            optimizer.zero_grad() # Clear gradients for next accumulation cycle
            scheduler.step()      # Update learning rate

            effective_step = global_step // args.gradient_accumulation_steps

            # --- Logging ---
            if (effective_step + 1) % args.logging_steps == 0:
                # Average metrics over logging steps * accumulation steps
                steps_since_last_log = args.logging_steps * args.gradient_accumulation_steps
                avg_loss = accumulated_loss / steps_since_last_log
                avg_lm_loss = accumulated_lm_loss / steps_since_last_log
                avg_balance_loss = accumulated_balance_loss / steps_since_last_log
                avg_supervised_loss = accumulated_supervised_loss / steps_since_last_log
                avg_router_reward = accumulated_router_reward / steps_since_last_log
                avg_attn_reward = accumulated_attn_reward / steps_since_last_log

                # Log metrics (Master Only)
                if is_master:
                    current_lr = scheduler.get_last_lr()[0]
                    log_msg = (
                        f"Step: {effective_step + 1}/{num_training_steps // args.gradient_accumulation_steps} | "
                        f"LR: {current_lr:.2e} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LM Loss: {avg_lm_loss:.4f} | "
                        f"Balance Loss: {avg_balance_loss:.4f} | "
                        f"Supervised Loss: {avg_supervised_loss:.4f} | "
                        f"Router Reward: {avg_router_reward:.2f} | "
                        f"Attn Reward: {avg_attn_reward:.2f} | "
                        f"Optim Step Time: {optimizer_step_duration:.3f}s"
                    )
                    logger.info(log_msg)
                    logger.info(f"TPU Metrics:\n{met.metrics_report()}") # Log XLA metrics

                    if args.use_wandb and wandb is not None:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lm_loss": avg_lm_loss,
                            "train/balance_loss": avg_balance_loss,
                            "train/supervised_loss": avg_supervised_loss,
                            "train/router_reward": avg_router_reward,
                            "train/attn_reward": avg_attn_reward,
                            "train/learning_rate": current_lr,
                            "train/epoch": (global_step * args.per_device_train_batch_size * world_size) / estimated_total_samples if 'estimated_total_samples' in locals() else effective_step / (steps_per_epoch_est // args.gradient_accumulation_steps), # Approximate epoch
                            "perf/optimizer_step_time": optimizer_step_duration
                            # Add TPU core utilization etc. if available from metrics
                        }, step=effective_step + 1)

                # Reset accumulators
                accumulated_loss = 0.0
                accumulated_lm_loss = 0.0
                accumulated_balance_loss = 0.0
                accumulated_supervised_loss = 0.0
                accumulated_router_reward = 0.0
                accumulated_attn_reward = 0.0


            # --- Saving Checkpoint ---
            if (effective_step + 1) % args.save_steps == 0:
                xm.master_print(f"Saving checkpoint at global step {global_step+1} (effective step {effective_step + 1})...")
                save_path = os.path.join(args.save_dir, f"step_{effective_step + 1}")

                # Save model, tokenizer, args (Master Only recommended for model saving)
                if is_master:
                    save_model_checkpoint(model, tokenizer, save_path, args)

                # Save optimizer/scheduler state using xm.save (handles distributed saving)
                # xm.save needs to be called on all processes
                # Use barrier=True to ensure all processes save before continuing
                xm.save({
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': effective_step + 1 # Save the effective step completed
                 }, os.path.join(save_path, "trainer_state.pt"), master_only=False) # Save on all replicas
                xm.master_print(f"Optimizer/Scheduler state saved by all processes to {save_path}")
                # Barrier after saving (optional, xm.save with barrier=True handles sync)
                # xm.rendezvous("checkpoint_save_done")


            # --- Evaluation Step (Optional) ---
            # Add evaluation logic here if eval_loader is configured
            # if eval_loader and (effective_step + 1) % args.eval_steps == 0:
            #     # Put model in eval mode
            #     # Run evaluation loop
            #     # Aggregate metrics across devices using xm.mesh_reduce
            #     # Log eval metrics
            #     pass


            # Check for max_steps completion
            if args.max_steps is not None and (effective_step + 1) >= (args.max_steps // args.gradient_accumulation_steps):
                 xm.master_print(f"Reached maximum effective steps ({effective_step + 1}). Stopping training.")
                 break # Exit the training loop

        # Increment global step counter AFTER potential optimizer step
        global_step += 1


    # === 8. Training Finished ===
    xm.master_print("Training finished.")
    # Save final model
    if is_master:
        xm.master_print("Saving final model...")
        final_save_path = os.path.join(args.save_dir, "final_model")
        save_model_checkpoint(model, tokenizer, final_save_path, args)
        xm.master_print(f"Final model saved to {final_save_path}")

    # Clean up WandB
    if is_master and args.use_wandb and wandb is not None:
        wandb.finish()

    xm.master_print("Training script finished on this core.")


# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XenArcAIi1 Model on TPU using PyTorch/XLA")

    # --- Add arguments to override TrainingArgs defaults ---
    # Example: Model Config Overrides (accessing nested dataclass fields)
    parser.add_argument("--dim", type=int, help=f"Override model dimension (default: {ModelArgs().dim})")
    parser.add_argument("--n_layers", type=int, help=f"Override number of layers (default: {ModelArgs().n_layers})")
    # Add other ModelArgs overrides as needed...

    # Example: Dataset and Training Hyperparameters
    parser.add_argument("--dataset_name", type=str, help="Hugging Face dataset identifier")
    parser.add_argument("--save_dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size per TPU core")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps (overrides epochs)")
    parser.add_argument("--train_max_seq_length", type=int, help="Maximum sequence length for training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--use_wandb", action='store_true', help="Enable W&B logging")
    parser.add_argument("--no_wandb", action='store_false', dest='use_wandb', help="Disable W&B logging")
    parser.set_defaults(use_wandb=True) # Default to using WandB if installed

    # Parse known args, allow unknown for flexibility if needed
    cli_args, unknown = parser.parse_known_args()

    # Create TrainingArgs instance and update with CLI overrides
    training_args = TrainingArgs()

    # Update ModelArgs nested dataclass if provided
    if cli_args.dim is not None: training_args.model_config.dim = cli_args.dim
    if cli_args.n_layers is not None: training_args.model_config.n_layers = cli_args.n_layers
    # Add updates for other ModelArgs fields...

    # Update TrainingArgs fields
    if cli_args.dataset_name is not None: training_args.dataset_name = cli_args.dataset_name
    if cli_args.save_dir is not None: training_args.save_dir = cli_args.save_dir
    if cli_args.learning_rate is not None: training_args.learning_rate = cli_args.learning_rate
    if cli_args.per_device_train_batch_size is not None: training_args.per_device_train_batch_size = cli_args.per_device_train_batch_size
    if cli_args.gradient_accumulation_steps is not None: training_args.gradient_accumulation_steps = cli_args.gradient_accumulation_steps
    if cli_args.num_train_epochs is not None: training_args.num_train_epochs = cli_args.num_train_epochs
    if cli_args.max_steps is not None: training_args.max_steps = cli_args.max_steps
    if cli_args.train_max_seq_length is not None: training_args.train_max_seq_length = cli_args.train_max_seq_length
    if cli_args.resume_from_checkpoint is not None: training_args.resume_from_checkpoint = cli_args.resume_from_checkpoint
    training_args.use_wandb = cli_args.use_wandb # Updated based on action flags


    # --- Start TPU Training ---
    # `xmp.spawn` will start the `_mp_fn` function on each available TPU core
    # It automatically sets environment variables for distributed communication
    logger.info("Spawning training processes across TPU cores...")
    xmp.spawn(_mp_fn, args=(training_args,), nprocs=None, start_method='fork')
    # nprocs=None automatically uses all available cores (usually 8 for v3-8, v4)
    logger.info("--- Training Complete ---")