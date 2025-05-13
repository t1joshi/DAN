# DAN

This repository contains the implementation and evaluation of Deep Averaging Network (DAN) models for sentiment classification on the Stanford Sentiment Treebank dataset.

## Project Structure

- `DANmodels.py` – Implements 2-layer and 3-layer DANs using:
  - Pre-trained GloVe embeddings (300d)
  - Randomly initialized embeddings

- `SUBWORDDANmodels.py` – Implements DANs using Byte Pair Encoding (BPE) with vocab sizes of 5k, 10k, and 30k.

- `main.py` – Training, testing, and experiment workflow using PyTorch.

## Models

| Architecture     | Embeddings        | Train Acc. | Dev Acc. |
|------------------|-------------------|------------|----------|
| DAN (2-layer)    | GloVe (300d)       | 98.9%      | 77.3%    |
| DAN (3-layer)    | GloVe (300d)       | 98.6%      | 78.1%    |
| DAN (2-layer)    | Random             | 99.3%      | 78.6%    |
| DAN (3-layer)    | Random             | 98.5%      | 77.8%    |
| DAN (2-layer)    | BPE (vocab 10k)    | 99.1%      | 77.6%    |
| DAN (2-layer)    | BPE (vocab 5k)     | 97.9%      | 77.4%    |
| DAN (2-layer)    | BPE (vocab 30k)    | 99.0%      | 73.4%    |

## Additional Work

- Includes a brief theoretical analysis of the Skip-Gram model and vector optimization for simple word-context pairs.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- SentencePiece (for BPE)

## Running the Code

To train a model:

```bash
python main.py --model N2NDAN --use_glove
