# 📊 Telco Customer Churn Prediction

This project analyzes customer churn behavior using data from a telecommunications company. It includes exploratory data analysis (EDA), feature engineering, and a predictive machine learning model to identify customers likely to churn. The goal is to support proactive retention strategies.

---

## 🔍 Problem Statement

Customer churn significantly affects the profitability of subscription-based businesses. Understanding which customers are likely to churn helps reduce revenue loss and improve customer retention. This project builds a data-driven model to predict churn from customer attributes and service usage patterns.

---

## 🧰 Tools & Technologies

- Python (Jupyter Notebook)
- pandas, NumPy, matplotlib, seaborn
- scikit-learn (Logistic Regression, Random Forest, KNN)
- Git & GitHub for version control and documentation
---

## 📁 Dataset

- **Source**: [IBM Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Size**: ~7,000 entries
- **Features**:
  - Demographics (gender, age, senior citizen)
  - Services (InternetService, Contract, PaymentMethod)
  - Tenure and monthly charges
  - Churn label (Yes/No)

---

## 🔄 Workflow Summary  
This project follows a typical machine learning pipeline:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training (Logistic Regression, Random Forest, KNN)
- Model Evaluation using Accuracy, Precision, Recall, F1 Score, PR AUC score and ROC AUC score.
- Insights & Recommendations

---

## 📊 Exploratory Data Analysis (EDA)

Key insights from the dataset:

- Customers on **month-to-month contracts** churn the most.
- **Electronic check** users have higher churn rates.
- **Shorter tenure** is strongly associated with churn.

### 📈 Sample Visualizations

#### Churn by Contract Type
![Churn by Contract](images/Churn_Contracts.png)

#### Correlation Heatmap
![Correlation Heatmap](images/Correlation_Map.png)

---

## 🔍 Key Insights

- **Contract Type** is the most significant churn predictor — customers on month-to-month contracts are much more likely to churn.
- **Tenure** and **Monthly Charges** also influence churn: newer customers and those with higher bills churn more frequently.
- Electronic check users had the highest churn rates among payment types.

---

## 📊 Model Performance Summary

- **Logistic Regression** showed the highest **recall**, making it suitable for use cases where catching all churners is vital.
- **Random Forest** offered the best balance between precision and recall (highest F1 Score).
- **KNN** underperformed in both precision and recall.

👉 _See the notebook for full model evaluation metrics, hyperparameter tuning, and visualizations._

---

## 📁 Project Structure

```
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── images/
│   └── [plots and visualizations]
├── churn_analysis.ipynb
├── README.md
└── requirements.txt
```

---

## ✅ How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook churn_analysis.ipynb
   ```

---

## 🎯 Future Improvements

- Automate data pipeline for real-time churn monitoring
- Integrate with a dashboard using Streamlit or Power BI
- Experiment with advanced models like XGBoost or LightGBM

---

## 📬 Contact

**Mahmood Arafat**  
Aspiring AI Engineer | Machine Learning Enthusiast

I'm passionate about building intelligent systems that solve real-world problems. With a background in Mechanical Engineering and hands-on experience in Python, data analysis, and model development, I'm actively transitioning into AI and machine learning—one project at a time.
[LinkedIn](https://www.linkedin.com/in/arafat-mahmood-3b0208213/) | [Email](Mahmoodarafat08@gmail.com)

---

