# Sentiment Analysis of Movie Reviews Using Machine Learning

A machine learning project that classifies IMDB movie reviews as **Positive** or **Negative** using multiple ML algorithms with custom feature engineering.

---

## Dataset

**IMDB Dataset of 50K Movie Reviews**  
- **Source:** [Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 balanced instances (25K positive / 25K negative)
- **Task:** Binary sentiment classification

---

## Feature Engineering

The raw text was engineered into 11 distinct attributes:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `text` | TF-IDF Vectorized review text |
| 2 | `rave_word_count` | Count of positive words (e.g., "masterpiece") |
| 3 | `rant_word_count` | Count of negative words (e.g., "garbage") |
| 4 | `technical_word_count` | Count of film terms (e.g., "cinematography") |
| 5 | `recommendation_word_count` | Count of recommendation phrases |
| 6 | `numerical_rating_mentioned` | Binary — whether a rating is mentioned |
| 7 | `word_count` | Total word count |
| 8 | `exclamation_count` | Number of exclamation marks |
| 9 | `question_count` | Number of question marks |
| 10 | `quote_count` | Number of quotes |
| 11 | `label` | Target: 0 = Negative, 1 = Positive |

---

## Models & Results

Nine models were trained and evaluated. **Logistic Regression** achieved the highest accuracy.

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| **Logistic Regression** | **88.57%** | 9.78s |
| LinearSVC | 88.29% | 3.01s |
| XGBoost | 85.59% | 44.50s |
| ExtraTrees | 83.80% | 0.83s |
| Random Forest | 83.23% | 1.44s |
| LightGBM | 81.37% | 7.91s |
| Gradient Boosting | 80.00% | 83.48s |
| AdaBoost | 77.14% | 28.56s |
| Decision Tree | 74.82% | 14.86s |

### Best Model — Logistic Regression

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.88 | 0.90 | 0.89 |
| Positive | 0.90 | 0.87 | 0.88 |

![Model Comparison Chart](model_comparison.png)

---

## Key Findings

- **Strongest negative predictor:** `rant_word_count` (correlation: -0.366)
- **Strongest positive predictor:** `rave_word_count` (correlation: +0.257)
- Linear models outperformed complex ensemble methods on this high-dimensional text task
- Hyperparameter tuning via Grid Search on Random Forest confirmed best params: `n_estimators=100`, `max_depth=20`, `min_samples_split=2`

---

## Methodology

1. **Preprocessing** — Remove HTML tags and special characters
2. **Split** — 80% training / 20% testing
3. **Vectorization** — TF-IDF for raw text
4. **Feature Extraction** — Statistical/semantic custom features
5. **Training** — 9 algorithms compared
6. **Tuning** — Grid Search on Random Forest

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```
