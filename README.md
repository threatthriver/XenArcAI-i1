# XenArc AI: Odysee Gen 1 - A Language Model

## Overview
XenArc AI is an open-source language model project that demonstrates the feasibility of training transformer-based models from scratch. The current prototype, Odysee Gen 1, is built on the XenArcAI architecture, designed for efficient training on TPUs and scalable deployment.

## Core Features
- **Advanced Transformer Architecture**
  - Transformer Encoder-based model with optimized layer configuration
  - 128k context length for extensive text processing
  - Character-level tokenization for granular text understanding
- **Performance Optimizations**
  - TPU-optimized training pipeline
  - Gradient clipping for stable training
  - Adaptive learning rate scheduler
  - Dropout regularization for better generalization
- **Production Ready**
  - Comprehensive validation pipeline
  - Checkpoint management system
  - Scalable inference capabilities

## Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (for GPU training)
- TPU access (for TPU training)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/XenArcAI-i1.git
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
  url={https://github.com/yourusername/XenArcAI-i1}
}
```
