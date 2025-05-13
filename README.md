# 💳 Predicting Credit Card Customer Defaults Using Machine Learning

## 📘 Project Overview

This project applies various **machine learning classification models** to predict whether a credit card customer will **default** on their payments. The dataset includes customer attributes such as age, credit limit, education, and past payment behaviors.

We evaluate and compare the performance of several models:
- **Decision Trees**
- **Bagging**
- **Random Forest**
- **Boosting**
- **K-Nearest Neighbors (KNN)**

---

## 🎯 Objective

To determine which algorithm provides the most accurate prediction of customer defaults, and to understand which features contribute the most to classification outcomes.

---

## 🧠 Machine Learning Pipeline

### 1. 🗂 Dataset Preparation
- **Target variable**: `default` (binary classification)
- **Features**: Demographic and financial attributes of credit card users
- **Preprocessing**:
  - Null value handling
  - One-hot encoding of categorical features
  - Basic statistics and feature review

### 2. 🧪 Models Implemented

#### ✅ Decision Tree
- Initial tree building
- Accuracy, confusion matrix, and ROC analysis
- Grid Search and pruning for model optimization

#### ✅ Bagging Classifier
- Reduces variance via ensemble of decision trees
- Performance measured with cross-validation and accuracy

#### ✅ Random Forest
- Uses multiple decision trees with randomized feature selection
- Feature importance visualization included

#### ✅ Boosting (e.g., AdaBoost / Gradient Boosting)
- Focuses on correcting misclassified instances
- Evaluated using confusion matrix and ROC AUC

#### ✅ K-Nearest Neighbors (KNN)
- Distance-based model with hyperparameter tuning (value of `k`)
- Compared based on classification metrics

---

## 📈 Evaluation Metrics

| Metric           | Description                                        |
|------------------|----------------------------------------------------|
| `Accuracy`       | Proportion of correct predictions                  |
| `Confusion Matrix` | True vs. predicted class breakdown              |
| `Precision/Recall` | To balance false positives and false negatives |
| `ROC Curve / AUC` | Overall classifier performance                    |

---

## 🧰 Technologies Used

| Tool / Library        | Purpose                                      |
|------------------------|----------------------------------------------|
| `Python`               | Programming language                         |
| `Pandas`, `NumPy`      | Data manipulation                           |
| `Matplotlib`, `Seaborn`| Data visualization                          |
| `Scikit-learn`         | Machine learning models, metrics, and tools  |

---

## 📊 Key Takeaways

- Ensemble methods like **Random Forest** and **Boosting** outperformed single models in terms of accuracy and generalization.
- **Feature importance analysis** revealed key drivers behind customer default behavior.
- Effective preprocessing and parameter tuning significantly impacted model performance.
