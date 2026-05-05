# Credit Card Fraud Detection Final Report

## Project Title
Predicting Fraudulent Credit Card Transactions

## Project Overview
This project studies whether machine learning can identify fraudulent credit card transactions from historical transaction records. The goal is to find patterns that separate fraudulent behavior from legitimate activity and to compare several classification models to determine which one offers the most practical balance between catching fraud and limiting false alarms.

This question matters because missed fraud leads to direct financial loss, while too many false alerts create customer friction and increase manual review costs. A useful fraud model therefore needs to do more than post a high accuracy score. It must perform well on metrics that matter in rare-event problems, especially precision, recall, F1-score, ROC-AUC, and average precision.

## Research Question
What factors influence whether a credit card transaction is fraudulent, and which machine learning model performs best at distinguishing fraudulent transactions from legitimate ones?

## Data Source
The analysis uses the public **Credit Card Fraud Detection** dataset from Kaggle. The dataset contains anonymized transaction records with the following fields:
- `Time`
- `Amount`
- transformed numerical variables `V1` through `V28`
- `Class`, where `0` represents a legitimate transaction and `1` represents fraud

## Technical Notebook
https://github.com/Chaitra013/Capstone/blob/main/credit_card_fraud_capstone.ipynb

## Approach
The project was completed in four main stages.

### 1. Data Preparation
- loaded the dataset
- reviewed the number of rows and columns
- checked data types and missing values
- identified and removed duplicate records

### 2. Feature Engineering
- created an `Hour` feature from transaction time
- created `Log_Amount` to reduce skew in transaction values
- created `Amount_Bin` to make transaction amount patterns easier to compare visually

### 3. Exploratory Data Analysis
- examined the class imbalance between fraudulent and legitimate transactions
- compared transaction amount distributions across classes
- reviewed time-based transaction patterns
- explored relationships between the fraud label and anonymized transformed variables

### 4. Model Building and Comparison
The final notebook compares four classification models:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

Model performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Training time

## Summary of Findings
The dataset is highly imbalanced, with fraudulent transactions making up only a very small portion of all records. That imbalance makes plain accuracy less useful, because a model can appear strong even if it misses important fraud cases.

The exploratory analysis showed that fraud does not depend on one obvious variable alone. Transaction amount and transaction timing provide some behavioral context, but several anonymized transformed variables appear to carry the strongest predictive signal. This suggests that fraud is shaped by a combination of patterns rather than a single rule.

The final model comparison showed clear differences in model behavior:
- **Random Forest** delivered the strongest overall balance of fraud-detection quality and ranking performance, with **precision = 0.8875**, **recall = 0.7474**, **F1-score = 0.8114**, **ROC-AUC = 0.9723**, and **Average Precision = 0.7765**.
- **KNN** produced the **highest precision (0.9412)**, which means it was the most conservative model and generated the fewest false alarms among the positive predictions it made, but its recall was lower at **0.6737**.
- **Logistic Regression** achieved the **highest recall (0.8737)**, meaning it detected the largest share of fraud cases, but it did so with very low precision (**0.0555**), which means many legitimate transactions were incorrectly flagged.
- **Decision Tree** reached recall of **0.8316**, but its precision (**0.0683**) and average precision (**0.4696**) showed that it was less reliable overall than Random Forest and KNN.

## Results
The results show that all models were able to detect some fraud, but they did not make the same kind of tradeoff.

Random Forest was the strongest overall model because it kept a high level of precision while still detecting a substantial portion of fraudulent transactions. In practical terms, it offered the best balance between catching fraud and avoiding too many unnecessary alerts.

KNN was even more selective than Random Forest. Its very high precision means that when it labeled a transaction as fraud, it was usually correct. The downside is that it missed more fraud cases than Logistic Regression.

Logistic Regression detected the most fraud overall, but it also generated many false positives. Its confusion matrix showed that it correctly identified **83 out of 95** fraud cases, but incorrectly flagged **1,412** legitimate transactions and still missed **12** fraudulent transactions. That makes it a useful benchmark for recall-focused detection, but less practical as a standalone model if false alerts are expensive.

Decision Tree performed better than Logistic Regression on accuracy, but its overall fraud-detection quality was weaker than Random Forest when considering the more important imbalanced-class metrics.

## Important Findings
Several practical insights came out of the final analysis:
- fraud detection should not be judged by accuracy alone in a rare-event dataset
- Random Forest provided the best overall balance across the most important metrics
- Logistic Regression is useful when the main priority is catching as many fraud cases as possible
- KNN is useful when the main priority is reducing false alarms
- the best production choice depends on business priorities, especially the cost of a missed fraud case versus the cost of reviewing a false alert

## Recommendations
Based on the final results, Random Forest is the strongest candidate for a practical fraud detection system in this dataset because it balances precision and recall more effectively than the other models tested.

The next improvements worth exploring are:
- tune decision thresholds instead of relying only on the default 0.5 cutoff
- apply class-imbalance strategies such as resampling, SMOTE, or class weighting
- test gradient-boosting models such as XGBoost or LightGBM
- evaluate cost-sensitive performance using business measures such as fraud losses avoided and review workload created
- explore feature-importance methods and model explainability tools to better understand why certain transactions are flagged

## Conclusion
This project shows that machine learning can meaningfully support credit card fraud detection even in a highly imbalanced dataset. The analysis confirmed that the dataset contains useful predictive patterns, but it also showed that model choice depends heavily on the type of error a business is more willing to tolerate.

Among the models tested, Random Forest produced the strongest overall results and emerged as the best balanced option. Logistic Regression was best at catching the highest share of fraud cases, while KNN was best at limiting false alerts. Taken together, these findings show that fraud detection is not only a modeling problem, but also a business decision about risk, customer experience, and operational cost.



