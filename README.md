Overview

This repository contains my implementation for the Language Modeling Lab (Lab 1). The goal of this assignment was to load a text dataset, tokenize it, group it into fixed-length sequences, and prepare it for training a causal language model.
In this version, I made specific targeted modifications to study how certain hyperparameters affect the preprocessing and training workflow.

Dataset

The dataset is loaded using the datasets library from HuggingFace. It provides a simple text corpus suitable for preparing training sequences for a language model.

Modifications Implemented

As part of customizing the assignment, I introduced three intentional changes to the model preparation pipeline.

1. Updated Sequence Length (Block Size)

Old Value: 128

New Value: 256

This allows the model to observe longer sequences at once, enabling it to learn longer contextual patterns in text.

2. Updated Batch Size

Old Value: 8

New Value: 16

A larger batch size provides more stable gradient updates and slightly improves training efficiency.

3. Updated Optimizer & Hyperparameters

I modified the training-related hyperparameters to explore their effect on optimization dynamics.

Parameter	Previous	Updated
Learning Rate	5e-4	3e-4
Optimizer	AdamW (default)	AdamW with weight_decay=0.01
Weight Decay	None	0.01
Warmup Steps	None	100
Epochs	Higher count	2

These updates introduce better regularization and smoother optimization behavior.

Pipeline Summary

The workflow includes:

Loading the dataset

Tokenizing text using a pretrained tokenizer

Grouping tokens into fixed-length sequences (block_size=256)

Creating padded training batches (batch_size=16)

Configuring an AdamW optimizer with modified hyperparameters

This setup supports an efficient and flexible preprocessing pipeline for language model training.

Results

The goal of these changes was not to maximize performance, but to better understand:

how longer sequences influence context learning

how larger batches affect stability

how optimizer modifications change training smoothness

The updated settings provided more stable training behavior and better regularization.

How to Run

Install necessary dependencies:

pip install transformers datasets torch accelerate


Run the notebook or script:

python train.py


or open the .ipynb file in Jupyter/Colab.

Conclusion

This modified version of the lab demonstrates how adjusting block size, batch size, and optimizer settings impacts the language model preprocessing workflow. The changes help reinforce important concepts around tokenization, batching, and hyperparameter configuration.
