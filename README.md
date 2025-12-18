# 🏠 House Prices Prediction – End-to-End Machine Learning Project

This project focuses on **predicting house prices in Ames, Iowa**, using advanced **regression techniques** and a complete **machine learning pipeline**, from data cleaning to model deployment with **Streamlit**.

It is designed as a **full Data Science project**, showcasing skills in data preprocessing, feature engineering, model selection, evaluation, and deployment.

---

## 🎯 Project Objectives

- Analyze and preprocess real-world housing data
- Engineer meaningful features for regression models
- Compare linear and non-linear regression approaches
- Evaluate model performance using appropriate metrics
- Deploy a trained model in an interactive **Streamlit web application**

--- 
## 🔬 Machine Learning Pipeline

###  Data Cleaning
- Handled missing values using context-aware imputation or removal
- Identified and removed outliers (e.g. abnormal living area vs price)
- Ensured data consistency and quality

--- 
###  Feature Engineering
- Creation and selection of relevant numerical and categorical features
- Categorical variables encoded using **One-Hot Encoding**
- Feature scaling applied where necessary 

---

###  Target Transformation
- The target variable **SalePrice** was log-transformed using `log1p`
- This reduces skewness and stabilizes variance, improving model performance

###  Model Training
Two regression models were trained and compared:

- **Ridge Regression**
  - Handles multicollinearity
  - Acts as a strong linear baseline

- **XGBoost Regressor**
  - Captures non-linear relationships
  - Models feature interactions effectively

---
##  Model Evaluation
- Models evaluated using:
  - **RMSE**
  - **R² Score**
- Learning curves used to diagnose:
  - Overfitting
  - Underfitting 


###  Deployment Preparation
- Final preprocessing objects and models saved using **joblib**
- Models integrated into a **Streamlit web application**
- Users can input house features and obtain real-time price predictions
