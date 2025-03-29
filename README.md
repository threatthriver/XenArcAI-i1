# BhaaratiAI: A Resource-Friendly Language Model for Indian Languages

## Core Features:

- Transformer Encoder-based Language Model
  - Transformer Encoder architecture for text generation.
- Character-level Tokenization
  - Simple character-level tokenizer for Indian languages.
- 128k Context Length
  - Limited context window for resource efficiency.
- Dropout Regularization
  - Dropout layer added to the Transformer Encoder for regularization.

## Prototype Implementation Plan:

- Data Collection & Preprocessing
  - Load data from a local text file (data/indian_text.txt).
  - Create a character-level tokenizer.
- Model Training
  - Train the Transformer Encoder model on the Indian text data.
  - Save model checkpoints during training.
- Text Generation
  - Generate text using the trained model.

## Final Goal:

BhaaratiAI aims to create a resource-friendly language model for Indian languages, demonstrating the feasibility of training such models from scratch.
