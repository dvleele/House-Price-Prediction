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
##  Tech Stack
- Python 3.13.5
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn (linear models, tree-based models, ensemble methods)  
- Docker (containerization)
- Render (deploy)

---
## Model performance comparison:

<img width="389" height="191" alt="image" src="https://github.com/user-attachments/assets/959464bb-b90c-4530-af56-b906ef7c7be8" />

**Best models**: Random Forest & Gradient Boosting  
**Worst model**: SVR (did not generalize well)

---
## API (FastAPI)
The trained model is wrapped inside a FastAPI service
---

## Docker
We containerized the API for easier deployment.

---
## Deployment (Render)

We deployed the Docker container to Render (free tier).
- Deployment service: Docker Web Service
- Platform: Render

---
## How to Test the API
- URL API: https://house-price-prediction-6c9b.onrender.com/docs
Steps to test:
1. Open the Swagger UI link above in your browser.
2. You will see all available endpoints.
3. Select POST /predict (the prediction endpoint).
4. Click "Try it out".
5. Enter sample JSON input (example below):
![](<Screenshot 2025-10-02 222725.png>)
![](<Screenshot 2025-10-02 222740.png>)

---
## Lessons Learned
- How to preprocess tabular data (missing values, feature selection)  
- Training and evaluating multiple regression models  
- Wrapping ML models into REST API with FastAPI
- Dockerizing ML applications for portability
- Deploying containerized apps to cloud platforms (Render)

---

- Author: Le Chi Dinh
- Purpose: Practice project for **Machine Learning portfolio**
