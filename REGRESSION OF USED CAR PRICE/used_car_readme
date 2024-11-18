## Project Overview
This project focuses on predicting the prices of used cars using machine learning techniques. The dataset, provided as part of the Kaggle Playground Series S4E9, is derived from a deep learning model trained on the original Used Car Price Prediction Dataset. The project involves preprocessing the data, feature engineering, training a LightGBM model, and optimizing its hyperparameters to achieve accurate predictions.

---

## Dataset Overview

### Files:
- **`train.csv`**: The training dataset with features and the target variable `price`.
- **`test.csv`**: The test dataset without the target variable `price`.
- **`sample_submission.csv`**: A sample submission file showing the required format for predictions.

### Target Variable:
- **`price`**: Continuous variable representing the price of the used car.

### Evaluation Metric:
The competition uses **Root Mean Squared Error (RMSE)** as the evaluation metric, defined as:

\\[
RMSE = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} \\left( y_i - \\hat{y}_i \\right)^2 }
\\]

where:
- \\( y_i \\): Actual value
- \\( \\hat{y}_i \\): Predicted value
- \\( N \\): Number of samples

---

## Project Workflow

### 1. **Data Loading**
The `train.csv` and `test.csv` files are loaded using `pandas`. The target variable `price` is separated from the features in the training dataset for further preprocessing.

---

### 2. **Exploratory Data Analysis (EDA)**
- **Missing Values**: Identified and handled missing values in both numerical and categorical columns.
- **Feature Distributions**: Examined the distributions of numerical variables using histograms and boxplots to detect skewness and outliers.
- **Log Transformation**: Applied log transformation to the `milage` column to address skewness.
- **Standardization**: Standardized numerical variables using MinMaxScaler to normalize their distributions.

---

### 3. **Feature Engineering**
- **Numerical Features**:
  - Log-transformed `milage` and standardized all numerical features.
- **Categorical Features**:
  - **Low Cardinality Encoding**: Used One-Hot Encoding for low-cardinality categorical variables such as `fuel_type`, `transmission`, and `brand`.
  - **High Cardinality Encoding**: Applied frequency encoding for high-cardinality categorical variables such as `model`, `engine`, `ext_col`, and `int_col`.
  - Imputed missing values using mode or custom logic where appropriate (e.g., `fuel_type` filled using the most frequent value within each `model` group).

---

### 4. **Data Preprocessing**
- Combined numerical and encoded categorical features into a single dataset.
- Split the data back into training and test sets, ensuring alignment with the original datasets.

---

### 5. **Modeling**
#### **Model Selection**
- Chose LightGBM (`LGBMRegressor`) for its efficiency and ability to handle large datasets and mixed feature types.

#### **Hyperparameter Tuning**
- Performed hyperparameter optimization using `RandomizedSearchCV` with a predefined search space for parameters like `n_estimators`, `learning_rate`, `max_depth`, etc.
- Used a scoring metric of negative mean squared error for tuning.

#### **Training and Evaluation**
- Trained the model on the training dataset.
- Evaluated the model using metrics:
  - **Mean Squared Error (MSE)**: To assess the average squared difference between predicted and actual values.
  - **R² Score**: To measure the proportion of variance in the target variable explained by the features.

---

### 6. **Prediction**
- Generated predictions for the test dataset using the optimized LightGBM model.
- Saved predictions in the required submission format (`id` and `predicted_price`) to `predicted_prices.csv`.

---

### 7. **Code Summary**
#### Key Libraries:
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Preprocessing**: `scikit-learn`
- **Modeling**: `lightgbm`

---

## Results
- Achieved strong performance on the training set with optimized hyperparameters.
- Generated predictions ready for submission to Kaggle.

---

## Author
**Matteo Omizzolo**  

