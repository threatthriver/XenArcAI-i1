# XenArc AI: Odysee Gen 1 - A Language Model

## Overview
XenArc AI is an advanced open-source language model project featuring a 600B parameter transformer-based architecture. The current prototype, Odysee Gen 1, demonstrates state-of-the-art capabilities in text generation and processing, with efficient training on TPUs and scalable deployment options.

## Model Architecture
- **Scale and Capacity**
  - 600B parameters for extensive knowledge representation
  - 48 transformer layers with 96 attention heads per layer
  - 12,288 hidden dimensions for rich feature extraction
  - 128k context length for comprehensive text understanding

## Core Features
- **Advanced Transformer Architecture**
  - Character-level tokenization (256 vocab size) for granular text understanding
  - Sinusoidal positional encoding for sequence awareness
  - 49,152 intermediate dimension in feed-forward networks
  - Multi-head self-attention with parallel computation

- **Performance Optimizations**
  - TPU-optimized training pipeline
  - AdamW optimizer with 0.1 weight decay
  - Gradient clipping at 1.0 for stable training
  - Cosine learning rate scheduling (1e-4 initial, 1e-5 minimum)
  - Mixed-precision training with gradient checkpointing

- **Production Ready**
  - Comprehensive validation pipeline
  - Advanced checkpoint management system
  - Scalable inference capabilities
  - Detailed metrics logging and analysis

## Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (for GPU training)
- TPU access (for TPU training)

### Installation
```bash
# Clone the repository
git clone https://github.com/threatthriver/XenArcAI-i1
cd XenArcAI-i1

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation
```python
# Download and preprocess training data
python download_and_preprocess.py --input_file data/input.txt
```

### Training
```python
# Train on GPU/CPU
python src/train.py --config config/model_600b.yaml

# Train on TPU
python src/tpu_train.py --config config/tpu_config.yaml
```

### Text Generation
```python
# Generate text using a trained model
python src/generate.py --model_checkpoint checkpoints/latest.pt --prompt "Your prompt here"
```

## Project Structure
```
.
├── config/              # Configuration files
├── docs/               # Detailed documentation
├── src/                # Source code
│   ├── data_pipeline.py  # Data processing
│   ├── model.py         # Model architecture
│   ├── train.py         # Training logic
│   └── generate.py      # Text generation
└── tests/              # Unit tests
```

## Configuration

Model and training parameters can be configured through YAML files in the `config/` directory:
- `model_600b.yaml`: Base model configuration
- `tpu_config.yaml`: TPU-specific training settings

## Contributing
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:
- Report issues
- Submit pull requests
- Set up development environment
- Follow code style guidelines

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to all contributors who have helped shape XenArcAI
- Special thanks to the open-source AI community

## Contact
For questions and support:
- Open an issue on GitHub
- Join our community discussions

## Citation
If you use XenArcAI in your research, please cite:
```bibtex
@software{xenarc_ai_2024,
  title={XenArc AI: Odysee Gen 1},
  author={XenArc AI Team},
  year={2024},
  url={https://github.com/threatthriver/XenArcAI-i1}
}
```
