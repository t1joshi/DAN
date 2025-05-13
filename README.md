# DAN
This repository contains the implementation and analysis of various Deep Averaging Network (DAN) models for sentiment classification using the Stanford Sentiment Treebank dataset.

📁 Contents
DANmodels.py: Implements two- and three-layer DANs using pre-trained GloVe embeddings and randomly initialized embeddings.

SUBWORDDANmodels.py: Implements DAN models using Byte Pair Encoding (BPE) subword tokenization with various vocabulary sizes.

main.py: Handles training, evaluation, and experiment orchestration.

🧠 Models
N2NDAN / N3NDAN: 2-layer and 3-layer DAN architectures.

Embedding Sources:

Pre-trained GloVe (300d)

Random initialization

BPE subwords (vocab sizes: 5k, 10k, 30k)

📊 Results Summary
Model Type	Train Accuracy	Dev Accuracy
DAN (GloVe, 2L)	98.9%	77.3%
DAN (GloVe, 3L)	98.6%	78.1%
DAN (Random, 2L)	99.3%	78.6%
DAN (Random, 3L)	98.5%	77.8%
DAN (BPE 10k, 2L)	99.1%	77.6%
DAN (BPE 5k, 2L)	97.9%	77.4%
DAN (BPE 30k, 2L)	99.0%	73.4%

🧪 Other Components
Skip-Gram model analysis to understand vector relationships and context probability estimation.

📦 Requirements
Python 3.8+

PyTorch

NumPy

SentencePiece (for BPE)

