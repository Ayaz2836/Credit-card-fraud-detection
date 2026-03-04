# Credit-card-fraud-detection
Credit card fraud detection decscription
This project provides a complete end-to-end pipeline for detecting fraudulent credit card transactions using Machine Learning. Based on the `creditcard.csv` dataset from Kaggle, the project addresses the critical challenge of **class imbalance** to build a reliable predictive model.

---

# Credit Card Fraud Detection System

## 📌 Project Overview

Credit card fraud detection is a classic binary classification problem. In a typical dataset, fraudulent transactions make up a tiny fraction (usually less than 0.2%) of the total data. This project demonstrates how to handle this imbalance, explore transaction patterns, and train a **Logistic Regression** model to identify suspicious activity.

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** * `pandas` & `numpy` (Data Manipulation)
* `seaborn` & `matplotlib` (Data Visualization)
* `scikit-learn` (Machine Learning)
* `kagglehub` (Dataset Acquisition)



---

## 🚀 Key Features of the Implementation

### 1. Data Acquisition & Cleaning

* **Automated Download:** Uses `kagglehub` to pull the latest version of the "mlg-ulb/creditcardfraud" dataset.
* **Integrity Check:** Performs a null-value analysis using `isnull().sum()` and visualizes it with a Seaborn heatmap to ensure no data gaps exist.

### 2. Handling Class Imbalance

The original dataset is highly skewed toward legitimate transactions. To prevent the model from simply "guessing" that every transaction is legitimate, the project implements **Under-sampling**:

* Identifies 492 fraud cases.
* Samples exactly 492 legitimate cases to create a balanced sub-dataset ($50/50$ distribution).
* Concatenates these into a `new_dataset` for unbiased training.

### 3. Exploratory Data Analysis (EDA)

The script generates several visualizations to understand the relationship between transaction features and fraud:

* **Correlation Heatmap:** To see which features (V1-V28) correlate most strongly with the `Class`.
* **Boxplots & Scatterplots:** To compare the `Amount` of fraud vs. legit transactions.
* **Class Distribution:** A count plot confirming the success of the balancing step.

---

## 📊 Model Performance

The project utilizes **Logistic Regression** for classification. Because accuracy can be misleading in fraud detection, the script calculates a comprehensive suite of metrics:

* **Accuracy Score:** Overall correctness.
* **Precision:** Ability to not label a legit transaction as fraud.
* **Recall:** Ability to find all fraudulent transactions (Crucial for banks).
* **F1-Score:** The harmonic mean of Precision and Recall.
* **Confusion Matrix:** A tabular summary of True Positives, True Negatives, False Positives, and False Negatives.

---

## 📂 How to Run

1. Ensure you have Python installed.
2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub

```


3. Run the script:
```bash
python credit_card_fraud_detection.py

```



---

## 💡 Future Improvements

* **Advanced Algorithms:** Implement Random Forest or XGBoost to capture non-linear patterns.
* **Over-sampling:** Try **SMOTE** (Synthetic Minority Over-sampling Technique) instead of under-sampling to retain more data.
* **Feature Scaling:** Apply `StandardScaler` to the `Time` and `Amount` columns for better convergence.

**Would you like me to help you modify the code to include SMOTE for better data retention?**
