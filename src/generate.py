import torch
from src.model import XenArcModel
from config.config import Config
from src.data_pipeline import TextDataset
import os

def generate(model, prompt, device, tokenizer, max_length=100, end_token_id=None):
    model.eval()
    tokens = []
    for c in prompt:
        if c in tokenizer.char_to_index:
            tokens.append(tokenizer.char_to_index[c])
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    generated_text = prompt
    with torch.no_grad():
        for _ in range(max_length):
            output = model(tokens)
            last_token_logits = output[ -1, :]
            predicted_token_id = torch.argmax(last_token_logits).item()
            if predicted_token_id in tokenizer.index_to_char:
                predicted_char = tokenizer.index_to_char[predicted_token_id]
                generated_text += predicted_char
                tokens = torch.cat((tokens, torch.tensor([[predicted_token_id]]).to(device)), dim=1)
            else:
                break
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
    model.load_state_dict(torch.load(model_path))
    prompt = "Hello, how are you?"
    generated_text = generate(model, prompt, device, tokenizer)
    print(generated_text)
