# House Price Prediction

## Overview
This project predicts house prices using the **Kaggle House Prices dataset**.  

We implemented and compared **8 regression models**:
- Linear Regression
- Ridge, Lasso, ElasticNet
- Support Vector Regression (SVR)
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

The goal is to practice an **end-to-end Machine Learning pipeline**:  
`data preprocessing → model training → evaluation → visualization`

---

## Dataset
- **Source**: [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
- **Shape**: `1460 rows × 81 columns`  
- **Target**: `SalePrice` (house price in USD)  

**Selected features (for baseline):**
- `OverallQual` – Overall material and finish quality  
- `GrLivArea` – Above ground living area (sq ft)  
- `GarageCars` – Number of cars garage can fit  
- `TotalBsmtSF` – Basement area (sq ft)  
- `FullBath` – Number of full bathrooms  

---

## ⚙️ Tech Stack
- Python 3.13.5
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn (linear models, tree-based models, ensemble methods)  

---

Model performance comparison:

<img width="389" height="191" alt="image" src="https://github.com/user-attachments/assets/959464bb-b90c-4530-af56-b906ef7c7be8" />

**Best models**: Random Forest & Gradient Boosting  
**Worst model**: SVR (did not generalize well)

---

## Lessons Learned
- How to preprocess tabular data (missing values, feature selection)  
- Training and evaluating multiple regression models  
- Understanding the difference between **linear vs non-linear models**  
- Ensemble methods (Random Forest, Gradient Boosting) often outperform single models  
- Importance of RMSE and R² in regression evaluation  

---

Author: Le Chi Dinh
Purpose: Practice project for **Machine Learning portfolio**
