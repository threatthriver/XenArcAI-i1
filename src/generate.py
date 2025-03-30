import torch
from src.model import XenArcModel
from config.config import Config
from src.data_pipeline import TextDataset
import os

def generate(model, prompt, device, tokenizer, config, max_length=None, temperature=1.0):
    """
    Generate text using the XenArcModel model.

    Args:
        model (nn.Module): The trained XenArcModel model.
        prompt (str): Initial text prompt.
        device (torch.device): Device to run generation on (CPU or CUDA).
        tokenizer (TextDataset): Tokenizer object from TextDataset.
        config (Config): Configuration object.
        max_length (int, optional): Maximum length of generated text. Defaults to None (uses config.max_length).
        temperature (float, optional): Sampling temperature. Lower values make generation more deterministic. Defaults to 1.0.

    Returns:
        str: Generated text.
    """
    model.eval()
    tokens = []
    for char in prompt:
        if char in tokenizer.char_to_index:
            tokens.append(tokenizer.char_to_index[char])
        else:
            print(f"Warning: Character '{char}' not in vocabulary. Skipping.") # Handle unknown characters
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    generated_text = prompt
    max_length = max_length if max_length is not None else config.max_length # Use config max_length if not provided

    with torch.no_grad():
        for _ in range(max_length):
            output = model(tokens)
            last_token_logits = output[:, -1, :] / temperature  # Apply temperature
            probabilities = torch.softmax(last_token_logits, dim=-1)
            predicted_token_id = torch.multinomial(probabilities, num_samples=1).item() # Sampling

            if predicted_token_id in tokenizer.index_to_char:
                predicted_char = tokenizer.index_to_char[predicted_token_id]
                generated_text += predicted_char
                tokens = torch.cat((tokens, torch.tensor([[predicted_token_id]], dtype=torch.long).to(device)), dim=1)
            else:
                break # Stop if token not in vocab (EOS or padding)
    return generated_text

if __name__ == '__main__':
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TextDataset("data", config)
    tokenizer = dataset.tokenizer
    vocab_size = tokenizer.vocab_size
    model = XenArcModel(config, vocab_size).to(device)
    # Load the trained model
    model_path = os.path.join(config.model_dir, "model_epoch_10.pth")
    model.load_state_dict(torch.load(model_path, map_location=device)) # Map to correct device
    prompt = "Hello, how are you?"
    generated_text = generate(model, prompt, device, tokenizer, config, temperature=0.8, max_length=50) # Added config and temp, max_length
    print(f"Generated text: {generated_text}") # More informative output
