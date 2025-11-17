📘 California Housing Price Prediction - End-to-End Machine Learning Project:

This repository contains a full end-to-end Machine Learning workflow built using Python and Scikit-Learn starting from raw data exploration all the way to model fine-tuning using GridSearchCV.
The project walks through real-world ML steps including data handling, EDA, feature engineering, pipelines, model evaluation, and hyperparameter tuning.


🗂️ Project Structure

├── data/                 # Housing dataset (after download)
├── notebooks/            # Jupyter notebooks with full workflow
└── README.md             # Project documentation (this file)

🚀 1. Project Overview

The goal of this project is to build a regression model that predicts median housing prices using the California Census dataset.
The workflow includes:
Downloading and understanding the dataset
Exploratory Data Analysis (EDA)
Handling missing values
Creating new features
Building preprocessing pipelines
Training ML models (Linear Regression, Decision Trees, Random Forests)
Improving evaluation with K-Fold cross-validation
Fine-tuning using GridSearchCV
Interpreting final results

📊 2. Exploratory Data Analysis (EDA)
Key insights:
Visualized geographic patterns using latitude/longitude scatter plots
Z-score normalization required for numeric stability
Strong correlations found between median income and house price
Identified categorical column (ocean_proximity) requiring one-hot encoding
Performed stratified sampling based on income categories to preserve distribution
EDA visuals are available in the images/ folder and the notebook.

🧩 3. Feature Engineering & Preprocessing
🔧 Feature Engineering
Implemented custom attributes such as:
rooms_per_household
population_per_household
bedrooms_per_room
These enhanced the predictive power of the model.
🛠️ Custom Transformer
Created a reusable Scikit-Learn transformer:
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    ...

This ensures clean integration in Pipelines and compatibility with GridSearchCV.
🔄 Preprocessing Pipeline
Used Pipeline and ColumnTransformer to automate:
Missing value imputation
Scaling (StandardScaler)
Feature engineering
One-hot encoding
Result: A fully prepared feature matrix ready for any model.

🤖 4. Model Training
Trained multiple models:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Initial evaluation used:
Train/Test Split RMSE
Overfitting/underfitting analysis

🧪 5. Model Evaluation — Cross-Validation
Used K-Fold Cross Validation (cv = 10) for reliable performance estimates.
Example:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)
Cross-validation prevented misleading results from a single train/test split and highlighted Decision Tree overfitting.

🔍 6. Model Fine-Tuning (GridSearchCV)
Performed hyperparameter tuning using:
param_grid = [
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]}
]
GridSearchCV tested 18 hyperparameter combinations × 5 CV folds = 90 model fits.
Random Forest achieved the best performance after tuning.

📈 7. Final Results
Final RMSE scores computed
Best hyperparameters selected using grid_search.best_params_
Best estimator exported for future predictions
The tuned model significantly outperformed baseline models.

📌 8. Technologies Used
Python
NumPy
Pandas
Matplotlib
Scikit-Learn
Jupyter Notebook

🙌 9. Acknowledgements
This project represents a complete real-world ML workflow demonstration, covering everything from raw data to a tuned regression model deployed through clean preprocessing pipelines.