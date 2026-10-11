# SentimentScope

## Movie Review Sentiment Analysis with a Transformer

SentimentScope is a natural language processing (NLP) project built with **PyTorch**. It trains a custom Transformer to automatically classify IMDB movie reviews by sentiment:

- `0` — negative review;
- `1` — positive review.

The project is provided as a Jupyter notebook covering the complete workflow: data loading, exploration, tokenization, batch preparation, architecture design, training, evaluation, and prediction on new reviews.

> **Reference result:** the notebook achieved a final test accuracy of **76.94%**, exceeding the project's 75% target.

## Table of Contents

- [Objectives](#objectives)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Configuration](#model-configuration)
- [Results](#results)
- [Predicting Sentiment](#predicting-sentiment)
- [Limitations and Possible Improvements](#limitations-and-possible-improvements)
- [References](#references)

## Objectives

This project demonstrates how to:

1. load and explore a text dataset;
2. prepare reviews for a deep learning model;
3. use a BERT-compatible Hugging Face tokenizer;
4. build a PyTorch `Dataset` / `DataLoader` pipeline;
5. adapt a Transformer architecture to binary classification;
6. train and evaluate a model on a GPU or CPU;
7. interpret the results and make predictions.

## Project Structure

```text
Project_02/
├── README.md
└── Sentiment_Scope_starter.ipynb
```

The `sentiment_model.pth` file is generated automatically by the notebook after achieving a test accuracy above 75%. It is not required to start the project.

## Dataset

This project uses the [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), which contains 50,000 movie reviews:

- 25,000 reviews for training;
- 25,000 reviews for testing;
- balanced positive and negative classes.

After extracting `aclImdb_v1.tar.gz`, the notebook expects the following structure next to `Sentiment_Scope_starter.ipynb`:

```text
aclImdb/
├── train/
│   ├── pos/
│   ├── neg/
│   └── unsup/       # not used
└── test/
    ├── pos/
    └── neg/
```

Files placed in `pos/` receive label `1`, while files placed in `neg/` receive label `0`.

## Prerequisites

- Python 3.10 or a compatible version;
- Jupyter Notebook or JupyterLab;
- an NVIDIA GPU with CUDA is recommended to reduce training time;
- enough disk space for the dataset and dependencies.

The notebook automatically detects the available hardware:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

A CPU can be used, but training the Transformer will take considerably longer.

## Installation

From the project directory, create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Activation on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activation on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install jupyterlab numpy pandas matplotlib torch transformers
```

> For an optimized CUDA installation, use the PyTorch command corresponding to your CUDA version from the [official PyTorch selector](https://pytorch.org/get-started/locally/).

Next, download the IMDB archive from [Stanford AI](https://ai.stanford.edu/~amaas/data/sentiment/), place `aclImdb_v1.tar.gz` in the project directory, and extract it:

```bash
tar -xzf aclImdb_v1.tar.gz
```

On Windows, the archive can also be extracted with 7-Zip or File Explorer. Make sure that the `aclImdb/` directory is located at the project root.

## Usage

Start Jupyter from the project root:

```bash
jupyter lab
```

Then open [`Sentiment_Scope_starter.ipynb`](./Sentiment_Scope_starter.ipynb) and run the cells from top to bottom.

The notebook:

1. extracts and loads the IMDB reviews;
2. builds the training and test DataFrames;
3. splits the training data into training and validation subsets;
4. initializes the `bert-base-uncased` tokenizer;
5. tokenizes, truncates, and pads each review to 128 tokens;
6. trains the Transformer;
7. calculates validation accuracy after each epoch;
8. measures final accuracy on the test set;
9. saves the weights to `sentiment_model.pth` if accuracy exceeds 75%.

During the first run, Hugging Face automatically downloads the files required by the `bert-base-uncased` tokenizer.

## How It Works

### Data Preparation

The `IMDBDataset` class converts each review into token IDs using the following settings:

```python
truncation=True
padding="max_length"
max_length=128
return_tensors="pt"
```

The data is then loaded in batches of 32 examples:

```python
BATCH_SIZE = 32
```

The notebook uses the following split:

| Subset | Number of examples |
|---|---:|
| Training | 22,500 |
| Validation | 2,500 |
| Test | 25,000 |

### Architecture

The `DemoGPT` model is a Transformer designed specifically for classification:

1. token and positional embeddings;
2. six Transformer blocks;
3. multi-head attention with four heads;
4. layer normalization;
5. mean pooling across the temporal dimension;
6. a linear layer producing two logits: negative or positive.

The tokenizer comes from `bert-base-uncased`, but the BERT model weights are not used. The project's Transformer architecture is trained directly for the classification task.

## Model Configuration

The main hyperparameters are:

| Parameter | Value |
|---|---:|
| Maximum sequence length | 128 tokens |
| Batch size | 32 |
| Embedding dimension | 256 |
| Number of Transformer blocks | 6 |
| Number of attention heads | 4 |
| Head dimension | 64 |
| Dropout | 0.1 |
| Number of classes | 2 |
| Number of epochs | 7 |
| Loss function | `CrossEntropyLoss` |
| Optimizer | `AdamW` |
| Learning rate | `3e-4` |

The configuration is grouped in the notebook's `config` dictionary. The `MAX_LENGTH`, `BATCH_SIZE`, and `EPOCHS` constants can be adjusted for experimentation, depending on the available memory.

## Results

The notebook includes a reference run with the following results:

- validation accuracy at the end of epoch 7: approximately 79%;
- final test accuracy: **76.94%**;
- project target: above **75%**;
- saved weights: `sentiment_model.pth`.

Accuracy may vary depending on the hardware, library versions, and the model's random initialization. No pretrained model is used for the Transformer's weights; only the BERT tokenizer is reused.

## Predicting Sentiment

After training, the `predict_sentiment` function accepts a review and returns `Positive` or `Negative`:

```python
review = "This movie was an absolute masterpiece! I loved it."
predict_sentiment(review, model, tokenizer, device)
```

Example output:

```text
Positive 🍿
```

The function puts the model in evaluation mode, applies exactly the same tokenization used during training, and selects the class with the highest logit.

## Limitations and Possible Improvements

This project is primarily educational. Possible next steps include:

- setting random seeds to improve reproducibility;
- recording and visualizing training metrics;
- using an attention mask to ignore padding tokens during mean pooling;
- adding cross-validation or hyperparameter search;
- comparing this custom-trained Transformer with a pretrained BERT model;
- saving the tokenizer and configuration alongside the model weights;
- exposing the model through an API or a demonstration interface.

## References

- [IMDB Large Movie Review Dataset — Stanford AI](https://ai.stanford.edu/~amaas/data/sentiment/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Dataset and DataLoader Tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Hugging Face Documentation — Padding and Truncation](https://huggingface.co/docs/transformers/pad_truncation)
- [`bert-base-uncased` Tokenizer](https://huggingface.co/bert-base-uncased)

## License and Attribution

This repository is an educational project completed as part of the Udacity program. The IMDB dataset remains subject to its own terms of use and attribution requirements.
