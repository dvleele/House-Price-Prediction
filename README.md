🏠 House Price Prediction
📌 Overview

This project predicts house prices using the Kaggle House Prices dataset.
We compare two models:
- Linear Regression (baseline)
- Random Forest Regressor (non-linear, ensemble method)
The goal is to practice an end-to-end Machine Learning pipeline:
data preprocessing → model training → evaluation → visualization.

📂 Dataset

Source: Kaggle – House Prices: Advanced Regression Techniques

Shape: 1460 rows × 81 columns

Target: SalePrice (house price in USD)

Selected features (for baseline):

- OverallQual – Overall material and finish quality

- GrLivArea – Above ground living area (sq ft)

- GarageCars – Number of cars that can fit in garage

- TotalBsmtSF – Basement area (sq ft)

- FullBath – Number of full bathrooms

⚙️ Tech Stack
- Python 3.15.5
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (LinearRegression, RandomForestRegressor)

📊 Results
Model	             |  RMSE   | 	R²
Linear Regression	 | ~40,051 |	0.79
Random Forest	     | ~29,478 | 0.89

✅ Random Forest outperforms Linear Regression.

📈 Visualizations
Predicted vs Actual Prices (Random Forest)

📚 Lessons Learned
- Handling missing values 
- Feature selection & encoding 
- Training and evaluating regression models 
- Comparing linear vs non-linear models
- Visualizing results
