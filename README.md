# Customer Churn Analysis and Prediction

A comprehensive machine learning project to analyze customer churn patterns and predict customer attrition using Python, SQL, and Scikit-learn. The project involves data preprocessing, exploratory analysis, feature engineering, model training, hyperparameter tuning, and evaluation.

---

## Project Overview

Customer churn prediction is crucial for businesses aiming to retain customers and reduce revenue loss. This project processes telecom customer data to build a predictive model that accurately identifies customers likely to churn. It helps stakeholders focus retention efforts efficiently.

---

## Key Features

* **Data Preprocessing:** Handles missing values, categorical encoding, and feature scaling.
* **Feature Engineering:** Creation of new features such as average charges per month.
* **Model Training:** Uses Logistic Regression, Random Forest, and Gradient Boosting with hyperparameter tuning.
* **Evaluation:** Metrics including Accuracy, Precision, Recall, F1-score, and ROC AUC.
* **Visualization:** Important feature ranking and ROC curve for model performance visualization.

---

## Output-->Screenshot Descriptions

### 1. Top 10 Feature Importance

The feature importance plot illustrates the top 10 variables that most influence the model's churn prediction. This ranking is derived from the Random Forest classifier's feature importance scores, highlighting which factors contribute most to customer attrition. Key observations:

* **Contract type** is the most significant factor, indicating customers with month-to-month contracts have a higher risk of churn.
* **Monthly Charges** and **Tenure** also play major roles, where higher charges and shorter tenure correlate with churn likelihood.
* **Payment Method** and **Internet Service Type** impact churn, reflecting customer preferences and service satisfaction.
* This insight helps business strategists prioritize retention efforts on critical customer segments.

<img width="998" alt="Screenshot 2025-05-26 at 1 26 09 AM" src="https://github.com/user-attachments/assets/4a4f2708-6523-4083-9369-453ea349da99" />


### 2. ROC Curve


The Receiver Operating Characteristic (ROC) curve measures the classifier’s ability to distinguish between churners and non-churners at various threshold settings:

* The **ROC AUC (Area Under Curve)** value of approximately **0.85** shows strong predictive power.
* The curve demonstrates a good balance of **True Positive Rate** (sensitivity) vs. **False Positive Rate**, indicating the model effectively identifies most churn cases while minimizing false alarms.
* This plot guides threshold selection for real-world deployment to optimize between precision and recall depending on business needs.

<img width="797" alt="Screenshot 2025-05-26 at 1 26 18 AM" src="https://github.com/user-attachments/assets/f4eb9a1e-cbca-4ad2-b7a3-e1ce79c91977" />

---

## How to Run

1. Clone the repo and navigate to the project root.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis script:

   ```bash
   python src/churn_analysis.py
   ```
4. View model results and visualizations.

---

## Dependencies

* Python 3.6+
* pandas, numpy, matplotlib, seaborn
* scikit-learn
* Jupyter Notebook (optional for exploratory analysis)

---

## Conclusion

This project delivers a robust churn prediction model backed by clear visualizations, providing actionable business insights. Feature importance guides targeted interventions, and the ROC curve validates model reliability. It serves as a practical tool for telecom companies to reduce churn and increase customer lifetime value.

