# Diabetes Prediction Project

## Project Overview

This project is dedicated to predicting diabetes using supervised machine learning models. 
We use a dataset containing medical predictor variables and one target variable, Outcome, which includes features like number of pregnancies, BMI, insulin level, and age.

## Libraries Used

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Seaborn & Matplotlib**: For data visualization.
- **Scipy**: For scientific computing.
- **Scikit-Learn**: For machine learning and predictive data analysis.

## Data Loading and Preprocessing

- **Source**: The dataset is loaded from a CSV file.
- **Inspection**: Basic data inspection for missing values and data types.
- **Details**: The dataset contains 768 entries and 9 columns with no missing values.

## Exploratory Data Analysis

- Box plots for visualizing data distribution.
- Z-score calculations for standardization.
- Identification and imputation of zero values in specific variables.
- Correlation analysis with heatmap visualization.
- Distribution analysis using histograms.
- Normality testing with Shapiro-Wilk test.
- Outcome-based group comparisons.

## Feature Engineering

- Imputation for variables like Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI.
- Detailed correlation analysis to understand feature-target relationships.

## Data Visualization

- Utilization of box plots, histograms, scatter plots, and pair plots for comprehensive data visualization.

## Model Preparation

- Data scaling using `StandardScaler`.
- Splitting data into training and testing sets.

## Model Training and Evaluation

- **Models Used**: Logistic Regression and Random Forest Classifier.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, and ROC AUC Score.

## Hyperparameter Tuning

- Conducting Grid Search for optimal parameters in Logistic Regression and Random Forest models.

## Cross-Validation

- Evaluating model performance using cross-validation techniques.

## Feature Importance Analysis

- Analyzing the importance of different features in the Random Forest model.

## Conclusion

- Final recommendations based on model performance considering various evaluation metrics.

## Repository Contents

- Detailed code, data, and findings are available in the Jupyter Notebook files within this repository.
