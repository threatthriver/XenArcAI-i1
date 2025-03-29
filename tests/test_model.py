import torch
from src.model import XenArcModel
from config.config import Config
from src.generate import generate

def test_model():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XenArcModel(config).to(device)
    prompt = "Hello, how are you?"
    generated_text = generate(model, prompt, device)
    assert prompt in generated_text

if __name__ == '__main__':
    test_model()
    print("Model test passed")
