# House Price Prediction — Stacking Ensemble

A machine learning pipeline for the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition. The model blends Ridge, Lasso, ElasticNet, and XGBoost regressors into a weighted ensemble, evaluated with 5-fold cross-validation RMSE on log-transformed sale prices.

---

## Project Structure

```
ames_housing.ipynb          # Main notebook (all steps below)
submission_improved.csv # Generated Kaggle submission file
```

---

## Pipeline Overview

### 1. Imports & Configuration
Loads all dependencies and reads the Kaggle train/test CSVs from `/kaggle/input/competitions/house-prices-advanced-regression-techniques/`.

### 2. Outlier Removal
Drops two well-documented anomalies: houses with `GrLivArea > 4000` sq ft but `SalePrice < $300,000`.

### 3. Target Transformation
Applies `log1p` to `SalePrice` to correct right skew and stabilise variance (heteroscedasticity). The target `y` is used in log scale throughout; predictions are back-transformed with `expm1` before submission.

### 4. Preprocessing
Train and test sets are concatenated for consistent encoding, then missing values are imputed:

| Strategy | Columns |
|---|---|
| Fill with `'None'` | Categorical features where NA means the feature is absent (e.g. `PoolQC`, `Alley`, `GarageType`) |
| Fill with `0` | Numeric features where NA means zero (e.g. `GarageArea`, `BsmtFinSF1`) |
| Median per Neighborhood | `LotFrontage` |
| Mode / Median | Remaining categoricals / numerics |

### 5. Feature Engineering
Twelve new features are derived from existing columns:

| Feature | Description |
|---|---|
| `TotalSF` | Basement + 1st floor + 2nd floor area |
| `TotalBathrooms` | Full + half baths (above and below grade) |
| `TotalPorchSF` | Sum of all porch area types |
| `HouseAge` | Year sold − year built |
| `YearsSinceRemodel` | Year sold − year remodelled |
| `IsNew` | 1 if built and sold in the same year |
| `WasRemodeled` | 1 if remodel year differs from build year |
| `HasPool` | 1 if pool area > 0 |
| `HasGarage` | 1 if garage area > 0 |
| `HasBsmt` | 1 if basement area > 0 |
| `HasFireplace` | 1 if fireplaces > 0 |

Numerical features with absolute skew > 0.75 are corrected with a Box-Cox transformation (`λ = 0.15`).

### 6. Encoding
One-hot encoding (`pd.get_dummies`, `drop_first=True`) is applied after engineering. The test matrix is re-indexed to match the training columns.

### 7. Model Training & Cross-Validation
All models are scored with **5-fold CV RMSE on log(SalePrice)**:

| Model | Notes |
|---|---|
| Ridge (α = 10) | Inside a `RobustScaler` pipeline |
| Lasso (α = 0.0004) | Inside a `RobustScaler` pipeline |
| ElasticNet (α = 0.0004, l1 = 0.9) | Inside a `RobustScaler` pipeline |
| XGBoost | 2000 trees, lr = 0.01, max_depth = 4, subsampling = 0.7 |

### 8. Lasso Feature Importance
After fitting, the top 20 Lasso coefficients are visualised (green = positive effect, red = negative effect) and the number of selected features is reported.

### 9. Stacking Ensemble
Weighted average of all four models (tuned by CV performance):

| Model | Weight |
|---|---|
| Ridge | 0.20 |
| Lasso | 0.20 |
| ElasticNet | 0.15 |
| XGBoost | 0.45 |

A custom `BlendedModel` sklearn estimator enables proper CV scoring of the ensemble.

### 10. Final Evaluation
In-sample metrics on the original dollar scale are reported: RMSE, MAE, and R². Three diagnostic plots are produced: Predicted vs Actual, Residuals vs Fitted, and a residual distribution histogram.

### 11. Kaggle Submission
Test predictions are back-transformed (`expm1`), clipped to zero, and saved to `submission_improved.csv`.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
statsmodels
scikit-learn
xgboost
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn xgboost
```

---

## Usage

This notebook is designed to run on **Kaggle** with the House Prices dataset already mounted. To run locally, update the data paths in Section 1:

```python
train = pd.read_csv('path/to/train.csv')
test  = pd.read_csv('path/to/test.csv')
```

Then run all cells in order. The final cell writes `submission_improved.csv` to the working directory.

---

## Reproducibility

A fixed random seed (`SEED = 42`) is used for all stochastic components (KFold shuffling, XGBoost, and sklearn models).
