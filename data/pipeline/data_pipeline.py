import jax
import jax.numpy as jnp
import webdataset as wds
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union, Iterator, Any
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from flax import jax_utils

class WebDataPipeline:
    def __init__(
        self,
        tokenizer_name: str,
        max_seq_length: int = 100000,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_buffer_size: int = 10000,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def preprocess_text(self, text: str) -> Dict[str, jnp.ndarray]:
        """Tokenize and prepare text for model input."""
        tokenized = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='jax'
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
    
    def filter_data(self, example: Dict) -> bool:
        """Filter low-quality or inappropriate content."""
        text = example.get('text')
        if not text or not isinstance(text, str):
            return False
        
        # Length filter (at least 10 words)
        words = text.strip().split()
        if len(words) < 10:
            return False
        
        # Basic quality checks
        if len(text.strip()) == 0 or text.isspace():
            return False
        
        return True
    
    def create_webdataset_pipeline(
        self,
        urls: Union[str, List[str]],
        filter_by_length: bool = True
    ) -> wds.DataPipeline:
        """Create a WebDataset pipeline for efficient streaming."""
        
        # Create dataset pipeline
        dataset = wds.WebDataset(urls)\
            .shuffle(self.shuffle_buffer_size)\
            .decode()\
            .map(json.loads)\
            .select(self.filter_data if filter_by_length else lambda x: True)\
            .map(self.preprocess_text)\
            .batched(self.batch_size)
        
        return dataset
    
    def create_jax_dataset(
        self,
        file_pattern: str,
        compression_type: Optional[str] = None
    ) -> Iterator[Dict[str, jnp.ndarray]]:
        """Create a JAX-compatible dataset for TPU training."""
        dataset = self.create_webdataset_pipeline(file_pattern)
        
        def prepare_batch(batch):
            # Convert to JAX arrays and shard across devices
            return jax_utils.replicate({
                'input_ids': jnp.array(batch['input_ids']),
                'attention_mask': jnp.array(batch['attention_mask'])
            })
        
        for batch in dataset:
            yield prepare_batch(batch)
    
    def process_large_dataset(
        self,
        input_files: List[str],
        output_path: str,
        num_shards: int = 1000
    ) -> None:
        """Process and shard large datasets for efficient training."""
        
        def process_shard(shard_id: int, files: List[str]):
            shard_examples = []
            for file in files:
                with open(file, 'r') as f:
                    for line in f:
                        example = json.loads(line)
                        if self.filter_data(example):
                            processed = self.preprocess_text(example['text'])
                            shard_examples.append(processed)
            
            # Save shard
            shard_path = f"{output_path}-{shard_id:05d}-of-{num_shards:05d}.tar"
            with wds.TarWriter(shard_path) as writer:
                for example in shard_examples:
                    writer.write({
                        '__key__': f'sample_{len(shard_examples)}',
                        'input_ids': example['input_ids'].tobytes(),
                        'attention_mask': example['attention_mask'].tobytes()
                    })
        
        # Distribute processing across shards
        files_per_shard = len(input_files) // num_shards
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for shard_id in range(num_shards):
                start_idx = shard_id * files_per_shard
                end_idx = start_idx + files_per_shard
                shard_files = input_files[start_idx:end_idx]
                executor.submit(process_shard, shard_id, shard_files)
        
        self.logger.info(f"Finished processing {len(input_files)} files into {num_shards} shards")