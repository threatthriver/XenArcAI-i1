import os
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
import subprocess

# Define datasets to download
text_datasets = [
    "pile", "bigbench", "gsm8k", "math_dataset",
    "arxiv", "semantic_scholar", "common_crawl"
]
code_datasets = ["bigcode/the-stack", "codeparrot/github-code"]
multi_modal_datasets = ["laion/laion-high-resolution", "webvid10m", "audiocaps", "shapenet"]

# Initialize tokenizer (GPT-4 level tokenization)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# Function to download and preprocess datasets
def download_and_process(dataset_name, split="train"):
    try:
        print(f"Downloading {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split)
        
        # Tokenization for text-based datasets
        if dataset_name in text_datasets or dataset_name in code_datasets:
            dataset = dataset.map(lambda x: {"tokens": tokenizer(x["text"])}, batched=True)
        
        # Save processed dataset
        save_path = f"/data/XenArc/datasets/{dataset_name}.jsonl"
        dataset.to_json(save_path)
        print(f"Saved {dataset_name} to {save_path}")

    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

# Process all datasets
for dataset in text_datasets + code_datasets + multi_modal_datasets:
    download_and_process(dataset)

print("All datasets downloaded and preprocessed successfully!")
