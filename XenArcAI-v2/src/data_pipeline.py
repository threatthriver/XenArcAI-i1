import jax
import jax.numpy as jnp
import tensorflow as tf
from transformers import AutoTokenizer
from datasets import load_dataset, IterableDataset
from typing import Dict, Iterator, Optional
import logging

from ..config.config import DataConfig # Use relative import

# Ensure TF does not allocate GPU memory. JAX handles device memory.
tf.config.experimental.set_visible_devices([], 'GPU')

logger = logging.getLogger(__name__)

def create_data_pipeline(config: DataConfig, split: str) -> tf.data.Dataset:
    """
    Creates a TensorFlow data pipeline for loading and processing data.

    Args:
        config: Data configuration object.
        split: Dataset split to load ('train' or 'validation').

    Returns:
        A tf.data.Dataset instance ready for training.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        # Set pad token if not present (common for GPT-2)
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Tokenizer pad token set to: {tokenizer.pad_token}")

    # --- Data Loading ---
    # IMPORTANT: This section uses a dummy dataset generator for demonstration.
    # Replace this with your actual data loading logic.
    # Examples:
    # 1. Loading from Hugging Face Hub:
    #    raw_dataset = load_dataset('your_dataset_name', split=split, streaming=True)
    #    raw_dataset = raw_dataset.to_tf_dataset(columns=['text']) # Adjust column name
    # 2. Loading from TFRecord files (matching config pattern):
    #    file_pattern = config.train_data_pattern if split == 'train' else config.eval_data_pattern
    #    filenames = tf.data.Dataset.list_files(file_pattern)
    #    raw_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
    #    # Add parsing function for your TFRecord structure here
    #    # raw_dataset = raw_dataset.map(parse_tfrecord_fn)
    logger.warning("Using dummy dataset generator for demonstration purposes. Replace with actual data loading.")
    if split == 'train':
        num_examples = 1000 # Dummy train size
        data_pattern = config.train_data_pattern # Use pattern from config
    else:
        num_examples = 100 # Dummy eval size
        data_pattern = config.eval_data_pattern # Use pattern from config

    # Create a dummy dataset generator
    def dummy_generator():
        for i in range(num_examples):
            yield {"text": f"This is dummy example number {i} for the {split} split."}

    # Create dataset from generator
    raw_dataset = tf.data.Dataset.from_generator(
        dummy_generator,
        output_signature={'text': tf.TensorSpec(shape=(), dtype=tf.string)}
    )
    # If using Hugging Face datasets:
    # raw_dataset = load_dataset('your_dataset_name', split=split, streaming=True) # Example
    # raw_dataset = raw_dataset.to_tf_dataset(columns=['text']) # Convert to tf.data

    # --- Tokenization ---
    def tokenize_function(examples: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        # tf.py_function requires eager execution context for numpy() and decode().
        # This tokenizes one example at a time.
        text = examples['text'].numpy().decode('utf-8')
        tokenized = tokenizer( # Tokenizer from Hugging Face transformers
            text,
            max_length=config.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf' # Return TensorFlow tensors for tf.data pipeline
        )
        # Ensure correct dtype for JAX/TPU compatibility later
        return {
            "input_ids": tf.cast(tokenized["input_ids"][0], tf.int32),
            "attention_mask": tf.cast(tokenized["attention_mask"][0], tf.int32),
        }

    # Use tf.py_function to wrap the tokenizer call
    def tf_tokenize_function(examples):
        return tf.py_function(
            func=tokenize_function,
            inp=[examples],
            Tout={
                "input_ids": tf.int32,
                "attention_mask": tf.int32,
            }
        )

    # Map the tokenization function. tf.py_function is used because the
    # tokenizer relies on Python logic (like string decoding and HF tokenizers)
    # that isn't directly convertible to TensorFlow graph operations.
    # num_parallel_calls enables parallel processing of examples.
    tokenized_dataset = raw_dataset.map(
        tf_tokenize_function,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # --- Batching and Prefetching ---
    if split == 'train':
        # Shuffle before batching for training data
        tokenized_dataset = tokenized_dataset.shuffle(
            config.shuffle_buffer_size, reshuffle_each_iteration=True
        )

    # Set shapes after py_function mapping
    tokenized_dataset = tokenized_dataset.map(
        lambda x: {
            "input_ids": tf.ensure_shape(x["input_ids"], [config.max_seq_length]),
            "attention_mask": tf.ensure_shape(x["attention_mask"], [config.max_seq_length]),
        }
    )

    # Batch examples. drop_remainder=True is crucial for TPUs, as they
    # expect fixed batch sizes for all devices.
    batched_dataset = tokenized_dataset.batch(
        config.batch_size_per_device,
        drop_remainder=True
    )

    # Prefetch data to overlap data loading/preprocessing with model execution.
    final_dataset = batched_dataset.prefetch(tf.data.AUTOTUNE)

    logger.info(f"Created data pipeline for split '{split}' using tokenizer '{config.tokenizer_name}' "
                f"with batch size {config.batch_size_per_device} per device.")
    return final_dataset

def get_dataset_iterator(dataset: tf.data.Dataset) -> Iterator[Dict[str, jnp.ndarray]]:
    """Converts a tf.data.Dataset to a Python iterator yielding JAX arrays."""
    for batch in dataset.as_numpy_iterator():
        # Convert TF tensors in the batch to JAX arrays
        yield jax.tree_map(jnp.asarray, batch)

if __name__ == '__main__':
    # Example usage (for testing the pipeline)
    logging.basicConfig(level=logging.INFO)
    dummy_data_config = DataConfig()

    print("Creating train dataset...")
    train_ds = create_data_pipeline(dummy_data_config, split='train')
    train_iter = get_dataset_iterator(train_ds)

    print("Fetching one batch from train dataset:")
    try:
        first_batch = next(train_iter)
        print("Batch keys:", first_batch.keys())
        print("Input IDs shape:", first_batch['input_ids'].shape)
        print("Attention Mask shape:", first_batch['attention_mask'].shape)
        print("Input IDs dtype:", first_batch['input_ids'].dtype)
    except StopIteration:
        print("Could not fetch a batch (dataset might be empty or too small).")

    print("\nCreating eval dataset...")
    eval_ds = create_data_pipeline(dummy_data_config, split='validation')
    eval_iter = get_dataset_iterator(eval_ds)

    print("Fetching one batch from eval dataset:")
    try:
        first_eval_batch = next(eval_iter)
        print("Batch keys:", first_eval_batch.keys())
        print("Input IDs shape:", first_eval_batch['input_ids'].shape)
    except StopIteration:
        print("Could not fetch an eval batch.")
