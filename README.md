# PlanetTerp-Review-Classifier
## Overview
This class project explores whether a transformation-based language model can **infer the star rating of a professor review based soly on review text**. Using reviews collected from PlanetTerp, I fine-tuned a pretrained **DistilBERT** model to predict how many stars (1-5) a reviewer assgned to a professor.  
The goal of the assignment was not to build a production-grade system, but to demonstrate understanding of **modern NLP workflows**, including data ingestion, transfomer fine-tuning, and model evaluation.
## Assignment Description
The assignment required building a tool that:
1. Ingests professor reviews using PlanetTerp API
2. Fine-tunes a PyTorch-based transformer model
3. Use the transformer to guess how many stars correspond to a review
4. Compares predicted rating to the actual ratings
## Project Goal
**Given the text of a professor review, predict the number of stars the reviewer gave.**
## Data Collection
Reviews were retrieved using the **PlanetTerp public API**
### Professor Selection Criteria
Professors were selected based on:
* Having 100+ reviews
* Having variations in ratings (not only 1-star or 5-star reviews)
### Professors Included
* Jonathan Fernades
* Kendall Williams
* Pendram Sadghiam
* Christiana Guest
* Monique Koppel
* James Rainbolt
* Cylde Kruskal
### Dataset Summary
* Total reviews: 920
* Features
  * 'prompt': Review text
  * 'label': Star rating (1-5)  
| **Rating** | **Count** | **Percentage** |
|:---:|:---:|:---:|
| 0 | 245 | 26.63% |
| 1 | 137 | 14.89% |
| 2 | 137 | 14.89% |
| 3 | 169 | 18.37% |
| 4 | 232 | 25.22% |  
The dataset is reasonably balanced, which makes it suitable for multi-class prediction without aggressive resampling

## Modeling Approach
### Data Splitting
The data was split using stratified sampling to preserve class balance:
* 70% Training
* 15% Validaiton
* 15% Test
Data was stored and managed using Hugging Face's `Dataset` and `DatasetDict` abstractions.

### Preprocessing
Text preprocessing was handled using HuggingFace tokenizers:
- **Tokenizer**: `distilbert-base-uncased`
- Tokenization and truncation to 128 tokens
- Padding handled dynamically during batching
- Professor identifiers were removed to avoid leakage


### Model
- **Architecture**:DistilBERT for Sequence Classification
- **Pretrained model**: `distilbert-base-uncased`
- **Number of output labels**: 5  
DistilBERT was chosen because it provides strong performance while remaining lightweight and suitable for a class project

### Training Setup
The model was fine-tuned using HuggingFace's `Trainer` API
#### Training Configuration
* Epochs: 3
* Learning rate: 5e-5
* Batch size: 16
* Weight deccay: 0.01
* Evaluation: per epoch
* Mixed precision (FP16) enabled  
The best model checkpoint (based on validation performance) was automaticaally loaded at the end of training.

## Evaluation
- Performance assessed using test set
### Evaluation Methods
  - Precision, Recall, and F1-score per class
  - Confusion matrix to visualize misclassifications
  - Manual inspection of incorrectly predicted examples  
The confusion matrix showed that most error occured between adjacent ratings (ex. 3 or 4), which is expected for subjective sentimental tasks.


## Error Analysis
Manual inspection of misclassifications examples revealed common failures modes often occurred in:
- Mixed sentimental reviews (positive and negative language in the same review)
- Longer reviews where the final rating depended on nuance or emphasis
- Reviews where numeric ratings conflicted with textual tone
### Example Misclassification
`"OK these reviews are way too dramatic lmao. These kids get to college and learn the hard way you can't BS your way through like high school, then they blame the professor when they fail. Rainbolt is a really underrated professor. I had him for Orgo I. … His lectures were great... My only complaint about this guy is he definitely is a bit full of himself and gets irritated when students ask questions/come to office hours. He's very impatient when it comes to that kind of stuff and I didn't like that. Other then that great professor."  
**Predicted**: 2  
**True**: 5
`

## Results & Observations
* The model performed reasonably well with identifying extreme ratings
* It struggled most with mid-range ratings (2-4)
* Longer reviews with mixed messaging were harder to classify accurately
* Overall, the model demonstrated that pretrained transformers can extract useful signals from student-written reviews, even with a relatively small dataset

## Tools & Technologies
- Python
- Pandas
- PyTorch
- Hugging Face Transformers

## Conclusion
While the model demonstrated strong sentiment recognition, predicting nuanced mid-range ratings remains challenging. This highlights inherent limitations of sentiment-based rating prediction.

## Reference
Much of the training and fine-tuning structure was adapted from:
* *Fine-Tuning BERT for Classification: A Practical Guide* - Hey Amit
* *Fine-Tuning DistilBERT: A Step-by-Step Practical Guide* - Hey Amit

