# Language Modeling Lab — Modified Version

## Overview
This repository contains my implementation for the Language Modeling Lab (Lab 1). The goal of this assignment was to load a text dataset, tokenize it, group the tokens into fixed-length sequences, and prepare them for a language-modeling task.

I introduced a few targeted modifications to better understand how certain hyperparameters affect preprocessing and model training behavior.

---

## Dataset
The dataset is loaded using the **HuggingFace `datasets` library** and provides a simple text corpus suitable for tokenization and sequence preparation.

---

## Modifications Implemented
Below are the exact updates I applied to customize the assignment.

---

### 1. Updated Sequence Length (Block Size)
- **Previous:** 128  
- **Updated:** **256**

This allows the model to receive longer text sequences in each training example, giving it more context to learn from.

---

### 2. Updated Batch Size
- **Previous:** 8  
- **Updated:** **16**

A bigger batch size improves training stability and makes better use of available memory.

---

### 3. Updated Optimizer and Training Hyperparameters
I modified several training parameters to evaluate how they influence optimization.

| Parameter        | Previous       | Updated                |
|------------------|----------------|-------------------------|
| Learning Rate    | 5e-4           | **3e-4**               |
| Optimizer        | AdamW (default) | **AdamW with weight decay** |
| Weight Decay     | None           | **0.01**               |
| Warmup Steps     | None           | **100**                |
| Epochs           | Higher count   | **2 epochs**           |

These adjustments help improve regularization and gradient smoothness during training.

---

## Pipeline Summary
The complete workflow includes:

1. Loading the dataset  
2. Tokenizing text using a pretrained tokenizer  
3. Splitting tokens into fixed-length sequences (`block_size = 256`)  
4. Preparing padded batches (`batch_size = 16`)  
5. Using AdamW with modified hyperparameters for training  

---

## Results
The goal of these changes was to explore how modifications to:

- sequence length  
- batch size  
- optimizer settings  

impact training dynamics.  
With the updated configuration, the training process became more stable and better regularized.

---

## How to Run
Install dependencies:

```bash
pip install transformers datasets torch accelerate
```

Run the notebook or script:

```bash
python train.py
```

or open the `.ipynb` notebook in Jupyter / Google Colab.

---

## Conclusion
This modified version of the lab highlights how adjusting core parameters such as block size, batch size, and optimizer settings can influence a language model’s training behavior. These changes make the pipeline more flexible and provide insight into how preprocessing affects model performance.

---


