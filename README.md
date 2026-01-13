# PlanetTerp-Review-Classifier


## Overview
This project builds an NLP pipeline to analyze professor reviews from PlanetTerp and predict sentiment and ratings using a fine-tuned transformer model. The goal is to assess how well textual sentiment aligns with numerical ratings provided by users.

## Motivation
Student reviews contain rich qualitative information that is not always captured by numeric ratings alone. This project explores whether modern transformer models can effectively learn sentiment signals and map them to rating scales.

## Data Collection
- Pulled professor reviews from PlanetTerp using a public API
- Each review includes:
  - Free-text feedback
  - User-assigned numerical rating

## Methodology
1. **Data Preprocessing**
   - Cleaned and structured review text
   - Tokenized text using Hugging Face tokenizers

2. **Modeling**
   - Fine-tuned a pretrained transformer model using PyTorch
   - Trained the model to perform sentiment / rating classification

3. **Evaluation**
   - Compared model predictions against user-provided ratings
   - Analyzed mismatches between textual sentiment and numeric scores

## Tools & Technologies
- Python
- Pandas
- PyTorch
- Hugging Face Transformers

## Results
The model demonstrated strong performance in capturing sentiment from review text, with observed gaps highlighting cases where written feedback and numeric ratings diverged.

## Report
A full technical report is included, detailing model architecture, training process, evaluation metrics, and limitations.
