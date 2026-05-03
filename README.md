# Car Price Prediction

A machine learning project that predicts the resale price of used cars based on features like fuel type, transmission, mileage, and age. Six regression models are trained and compared, with the best model tuned using GridSearchCV.

---

## Dataset

**Source:** `car_data.csv` (301 records, 9 features)

| Column | Description |
|--------|-------------|
| `Car_Name` | Name of the car |
| `Year` | Manufacturing year |
| `Selling_Price` | Resale price in Lakhs *(target)* |
| `Present_Price` | Current ex-showroom price in Lakhs |
| `Driven_kms` | Total kilometers driven |
| `Fuel_Type` | Petrol / Diesel / CNG |
| `Selling_type` | Dealer / Individual |
| `Transmission` | Manual / Automatic |
| `Owner` | Number of previous owners |

---

## Project Structure

```
car-price-prediction/
│
├── data/
│   └── car_data.csv
│
├── notebooks/
│   └── car_price_prediction.ipynb
│
├── src/
│   └── train.py
│
├── results/
│   ├── selling_price_distribution.png
│   ├── categorical_vs_price.png
│   ├── scatter_plots.png
│   ├── price_by_year.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── cross_validation.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Workflow

1. **EDA** — Distribution plots, scatter plots, correlation heatmap, price trends by year
2. **Feature Engineering** — Created `Car_Age`, `Depreciation_Ratio`, `KM_Per_Year`
3. **Preprocessing** — Label encoding for categorical columns, StandardScaler for linear models, 80/20 train-test split
4. **Model Training** — Six models trained and evaluated
5. **Evaluation** — R², RMSE, MAE, and 5-fold cross-validation
6. **Tuning** — GridSearchCV on Random Forest
7. **Feature Importance** — Ranked by the best Random Forest model

---

## Models Used

| Model | Type |
|-------|------|
| Linear Regression | Linear |
| Ridge Regression | Regularized Linear |
| Lasso Regression | Regularized Linear |
| Decision Tree | Tree-based |
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |

---

## Results

Random Forest and Gradient Boosting consistently outperform linear models on this dataset. `Present_Price` and `Car_Age` are the most influential features.

---

## Setup

```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
pip install -r requirements.txt
```

Open the notebook:
```bash
jupyter notebook notebooks/car_price_prediction.ipynb
```

Or run the script directly:
```bash
python src/train.py
```

> **Google Colab users:** Upload `car_data.csv` to the Files panel and change the data path in the notebook to `'car_data.csv'`.

---

## Technologies

- Python 3.10
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
