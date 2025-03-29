# Odysee Gen 1: A Resource-Friendly Language Model for Indian Languages

## Overview

Odysee Gen 1 is a prototype language model designed for Indian languages. It aims to be resource-friendly, allowing for training and deployment on limited hardware. The model is trained from scratch without using any pre-trained models.

## Core Features

- Transformer Encoder-based Language Model
- Character-level Tokenization
- 128k Context Length
- Dropout Regularization
- Positional Encoding

## Model Architecture

The model consists of the following layers:

1.  **Embedding Layer:** Maps character indices to embedding vectors.
2.  **Positional Encoding Layer:** Adds positional information to the embedding vectors.
3.  **Transformer Encoder Layer 1:** Processes the embedded input using a Transformer encoder layer.
4.  **Transformer Encoder Layer 2:** Processes the output of the first Transformer encoder layer using a second Transformer encoder layer.
5.  **Linear Layer:** Maps the output of the second Transformer encoder layer to the vocabulary size.

The Transformer encoder layers use multi-head attention and feedforward networks to capture long-range dependencies in the input sequence. Positional encoding is added to the input embeddings to provide information about the position of each token in the sequence.

## Training

The model is trained using the following steps:

1.  Load data from a local text file (`data/indian_text.txt`).
2.  Create a character-level tokenizer.
3.  Train the Transformer Encoder model on the Indian text data.
4.  Save model checkpoints during training.

## Usage

The model can be used to generate text by providing a prompt. The generated text will be in the style of the Indian text data used to train the model.

## Limitations

-   The model is trained on a small dataset, so the generated text may not be very coherent.
-   The model is a prototype, so it may not be suitable for production use.

## Future Work

-   Train the model on a larger dataset.
-   Explore different model architectures.
-   Implement more advanced training techniques.
