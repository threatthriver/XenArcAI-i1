import torch
import os
from torch.utils.data import Dataset, DataLoader
from config.config import Config

class TextDataset(Dataset):
    def __init__(self, data_dir, config):
        self.config = config
        self.data_dir = data_dir
        self.data = self.load_data()
        self.tokenizer = self.create_tokenizer()

    def load_data(self):
        """
        Load text data from the specified file.

        Returns:
            str: The loaded text data.
            list: Empty list if file not found.
        """
        file_path = os.path.join(self.data_dir, "indian_text.txt")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except UnicodeDecodeError:
            print(f"UnicodeDecodeError: Unable to read {file_path} with utf-8 encoding. Please check file encoding.")
            return []
        return text

    def create_tokenizer(self):
        """
        Create character-level tokenizer.

        Returns:
            TextDataset: The tokenizer object (self).
        """
        chars = sorted(list(set(self.data)))
        self.char_to_index = {ch: i for i, ch in enumerate(chars)}
        self.index_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        return self

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return max(0, len(self.data) - self.config.context_length)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: Input tensor for training.
        """
        chunk = self.data[idx:idx + self.config.context_length]
        input_indices = [self.char_to_index[ch] for ch in chunk]
        input_tensor = torch.tensor(input_indices, dtype=torch.long)  # Explicitly set dtype to long
        return input_tensor

if __name__ == '__main__':
    config = Config()
    dataset = TextDataset("data", config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    for batch in dataloader:
        print(batch.shape)
        break
