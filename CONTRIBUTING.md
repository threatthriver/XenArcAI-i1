# Contributing to XenArcAI

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch latest stable version
- CUDA-compatible GPU (for training)

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/XenArcAI-i1.git
   cd XenArcAI-i1
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Testing
- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Run tests using:
  ```bash
  python -m pytest tests/
  ```

### Commit Messages
- Use clear, descriptive commit messages
- Format: `[Component] Brief description`
- Example: `[Model] Add gradient checkpointing support`

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and commit them
3. Push to your fork and submit a pull request
4. Ensure PR description clearly describes the changes
5. Include any relevant issue numbers

### PR Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Changes are focused and atomic

## Model Development

### Architecture Changes
- Document parameter changes in config files
- Update architecture documentation
- Validate performance impact
- Consider backward compatibility

### Training Modifications
- Document hyperparameter changes
- Provide training curves and metrics
- Include resource requirements

## Questions or Issues?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
- General questions

Thank you for contributing to XenArcAI!