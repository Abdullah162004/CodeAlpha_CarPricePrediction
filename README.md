# Car Price Prediction using Machine Learning

**CodeAlpha Data Science Internship — Task 3**

## Overview

This project predicts the resale price of used cars based on features like brand, manufacturing year, fuel type, transmission, kilometers driven, and ownership history. It covers the full machine learning pipeline from data cleaning and feature engineering to training, evaluating, and tuning multiple regression models.

## Dataset

**car data.csv** — 301 records with 9 columns:

| Column        | Description                                      |
|---------------|--------------------------------------------------|
| Car_Name      | Name/model of the car                            |
| Year          | Year of manufacture                              |
| Selling_Price | Price at which the car is being sold (Lakhs INR) |
| Present_Price | Current ex-showroom price of the car (Lakhs INR) |
| Driven_kms    | Total kilometers driven                          |
| Fuel_Type     | Fuel type — Petrol, Diesel, or CNG               |
| Selling_type  | Seller type — Dealer or Individual               |
| Transmission  | Transmission type — Manual or Automatic          |
| Owner         | Number of previous owners (0, 1, 3)              |

## Project Structure

```
CodeAlpha_CarPricePrediction/
├── car_price_prediction_CodeAlpha.ipynb   # Main Colab notebook
├── car data.csv                           # Dataset
└── README.md
```

## Steps Performed

1. **Import Libraries** — Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
2. **Load Dataset** — 301 rows, 9 features including target (Selling_Price)
3. **Exploratory Data Analysis**
   - Distribution plots for all numerical features
   - Average selling price by Fuel Type, Seller Type, and Transmission
   - Scatter plots of numeric features vs Selling Price
   - Correlation heatmap
4. **Feature Engineering**
   - `Car_Age` — derived from manufacturing year
   - `Depreciation_Ratio` — Selling Price / Present Price
   - `KM_Per_Year` — Driven_kms / Car_Age
   - Label encoding for categorical columns (Fuel_Type, Selling_type, Transmission)
   - Dropped Car_Name and Year after feature extraction
5. **Data Preparation** — 80/20 train/test split, StandardScaler for linear models
6. **Model Training** — Six models trained and evaluated
7. **Model Evaluation** — MAE, RMSE, R2, and 5-fold cross-validation R2 for all models
8. **Hyperparameter Tuning** — GridSearchCV on the best performing Random Forest model
9. **Feature Importance** — Ranked feature importance from the tuned Random Forest
10. **10-Fold Cross Validation** — Stability check on the final tuned model

## Models Used

| Model              | R2     | RMSE  | MAE   |
|--------------------|--------|-------|-------|
| Gradient Boosting  | 0.9879 | 0.528 | 0.302 |
| Random Forest      | 0.9827 | 0.632 | 0.381 |
| Decision Tree      | 0.9600 | 0.959 | 0.555 |
| Linear Regression  | 0.8814 | 1.653 | 1.066 |
| Ridge Regression   | 0.8804 | 1.660 | 1.068 |
| Lasso Regression   | 0.8799 | 1.663 | 1.070 |

**Best Model: Gradient Boosting — R2 = 0.9879**

## Feature Importance (Random Forest)

| Feature            | Importance |
|--------------------|------------|
| Present_Price      | 88.3%      |
| Depreciation_Ratio | 7.7%       |
| Car_Age            | 1.4%       |
| Driven_kms         | 1.1%       |
| Transmission       | 0.7%       |
| KM_Per_Year        | 0.5%       |
| Selling_type       | 0.2%       |
| Owner              | 0.1%       |
| Fuel_Type          | 0.0%       |

## How to Run

1. Open `car_price_prediction_CodeAlpha.ipynb` in Google Colab or Jupyter.
2. Upload `car data.csv` to the Colab session files (or place it in the same directory).
3. Run all cells sequentially from top to bottom.

**Required libraries** (all pre-installed in Colab):
```
numpy, pandas, matplotlib, seaborn, scikit-learn
```

## Key Findings

- Present Price (current ex-showroom value) is by far the most influential feature, accounting for 88% of the model's decision making.
- The engineered Depreciation Ratio is the second most important feature at 7.7%, confirming that how much a car has depreciated relative to its original price matters significantly.
- Car Age is a stronger predictor than raw kilometers driven.
- Automatic transmission cars tend to have higher average resale prices than manual ones.
- Diesel cars command a higher resale value on average compared to Petrol and CNG.
- Ensemble methods (Gradient Boosting, Random Forest) significantly outperform linear models on this dataset.

## Tools & Libraries

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Author

CodeAlpha Data Science Internship
