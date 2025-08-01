# Model Performance Results

This document summarizes the performance of the latest version of the trading model.

## Training Details

- **Data:** 20,000 candles per symbol
- **Features:** 13 features, including price action, technical indicators, and session information.
- **Model:** XGBoost Classifier with hyperparameter tuning (`GridSearchCV`)
- **Best Parameters:** `{'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}`
- **Best F1 Score (weighted):** `0.5447`

## Evaluation on Test Set

### Classification Report

```
              precision    recall  f1-score   support

    Loss (0)       0.80      0.59      0.68     17893
 TP1 Win (1)       0.20      0.31      0.24      3701
 TP2 Win (2)       0.32      0.45      0.37      6226

    accuracy                           0.52     27820
   macro avg       0.44      0.45      0.43     27820
weighted avg       0.61      0.52      0.55     27820
```

### Confusion Matrix

```
[[10526  2978  4389]
 [  955  1154  1592]
 [ 1641  1769  2816]]
```

## Analysis

The model's performance has improved significantly with the larger dataset. Here are the key takeaways:

- **Improved Accuracy:** The overall accuracy has increased to 52%, which is a positive sign.
- **Better Win Prediction:** While still not ideal, the precision and recall for winning trades (`TP1 Win` and `TP2 Win`) have improved. This means the model is getting better at identifying profitable opportunities.
- **Strong Loss Prediction:** The model continues to be effective at predicting losses, with a precision of 80%. This is valuable for risk management.

While the model is not yet consistently profitable, these results are very promising. The next steps should focus on further improving the precision of the win predictions. This could involve:

- **More advanced feature engineering:** Exploring new features or combinations of features.
- **More extensive hyperparameter tuning:** Using a larger search space for `GridSearchCV`.
- **Trying different models:** Experimenting with other algorithms like LightGBM or deep learning models.
