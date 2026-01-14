# PlanetTerp-Review-Classifier
## Overview
This project builds a transformer-based NLP model to predict professor star ratings from textual reviews collected from PlanetTerp. The goal is to evaluate how effectively sentiment and contextual cues in reviews map to numeric ratings.

## Problem Statement
Given a written professor review, can a fine-tuned transformer model accurately predict the reviewer’s star rating?

## Data Collection
- Reviews pulled using the **PlanetTerp public API**
- Dataset fields:
  - Review text
  - User-provided star rating
- Professors selected to ensure sufficient review counts and reduce extreme class imbalance

## Modeling Approach
### Preprocessing
- Cleaned and structured review text
- Tokenized inputs using Hugging Face tokenizers

### Model
- Fine-tuned **DistilBERT** for multi-class classification
- Trained using PyTorch and Hugging Face’s Trainer API

### Experimentation
- Iterated on:
  - Dataset composition to improve class balance
  - Learning rates (5e-5 → 1e-5)
- Evaluated training and validation loss trends to detect overfitting

## Evaluation
- Performance assessed using:
  - Validation loss
  - F1 score
  - Confusion matrix analysis
- Model performed best on extreme ratings but struggled with **mid-range and mixed-sentiment reviews**

## Error Analysis
Misclassifications often occurred in:
- Longer reviews with both positive and negative sentiments
- Reviews where numeric ratings conflicted with textual tone

## Tools & Technologies
- Python
- Pandas
- PyTorch
- Hugging Face Transformers

## Conclusion
While the model demonstrated strong sentiment recognition, predicting nuanced mid-range ratings remains challenging. This highlights inherent limitations of sentiment-based rating prediction.

## Report
A detailed technical report describing model architecture, training strategy, and evaluation results is included.

