# News Article Text Classification

A machine learning project that automatically classifies news articles into categories using Logistic Regression and BERT. Built using the AG News dataset with 127,600 articles.

---

## Problem Statement

The massive volume of online news makes manual sorting slow, inconsistent, and error-prone. This project automates news classification to:
- Reduce manual effort and human errors
- Enable personalized news recommendations
- Improve search and retrieval of relevant articles

---

## Dataset

**AG News Dataset**

| Column | Description |
|---|---|
| `title` | Headline of the news article |
| `description` | Brief summary of the article |
| `category` | News category label |

**Categories:** World, Sports, Business, Science / Technology

**Size:** 127,600 news articles · 3 columns

---

## Models

### Logistic Regression
- Traditional classifier — simple, interpretable, robust
- Text preprocessing: tokenization, stop-word removal, lemmatization
- Vectorization: TF-IDF

**Pipeline:**
1. Tokenization using TF-IDF
2. Train/test split
3. Initialize weights and biases
4. Compute weighted sum `y = wx + b`
5. Softmax activation
6. Compute loss
7. Update weights and biases
8. Evaluation

---

### BERT (Fine-tuned)
- Bidirectional Encoder Representations from Transformers
- Captures complex contextual meaning in text

**Pipeline:**
1. Tokenization
2. Convert tokens to IDs
3. Pass through Transformer layers
4. Classification head
5. Training
6. Prediction

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.87 | 0.87 | 0.87 | 0.87 |
| BERT (Fine-tuned) | 0.94 | 0.94 | 0.94 | 0.94 |

---

## Workflow

```
1. Data Collection
2. Data Preprocessing
3. Model Training
4. Model Evaluation
5. Prediction
```

---

## Project Structure

```
News-Classification/
├── BERT.ipynb                  # BERT fine-tuning and classification
├── Logistic Regression.ipynb   # Logistic regression with TF-IDF
├── .gitignore
└── README.md
```

---

## Tech Stack

- **Language:** Python
- **ML Models:** Logistic Regression, BERT (HuggingFace Transformers)
- **NLP:** TF-IDF, Tokenization, Stop-word removal, Lemmatization
- **Libraries:** Scikit-learn, PyTorch, Transformers, Pandas, NumPy

