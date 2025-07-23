# Loan Approval Prediction

This project predicts whether a loan application will be approved or not using machine learning techniques such as Logistic Regression and Random Forest. It includes full data preprocessing, feature engineering, model training, evaluation, and final prediction on unseen test data.

## Dataset

The dataset is divided into:
- `train.csv` – labeled data with loan status.
- `test.csv` – unlabeled data for prediction.

## Key Features

- **Gender** – Applicant's gender (`Male` / `Female`)
- **Married** – Marital status (`Yes` / `No`)
- **Dependents** – Number of dependents (`0`, `1`, `2`, `3+`)
- **Education** – Educational qualification (`Graduate` / `Not Graduate`)
- **Self_Employed** – Employment status (`Yes` / `No`)
- **ApplicantIncome** – Applicant's monthly income
- **CoapplicantIncome** – Co-applicant’s monthly income
- **LoanAmount** – Loan amount in thousands
- **Loan_Amount_Term** – Term of loan in months
- **Credit_History** – Whether the applicant has credit history (`1` / `0`)
- **Property_Area** – Location of the property (`Urban`, `Semiurban`, `Rural`) 


## Problem Statement

Predict the binary loan approval status (`Loan_Status`):
- `Y` – Approved
- `N` – Not Approved

## Workflow

### 1. Data Loading
- Load `train.csv` and `test.csv` using pandas.
- Append a placeholder `Loan_Status` column to `test.csv` and concatenate both datasets for uniform preprocessing.

### 2. Data Cleaning & Preprocessing
- **Missing Value Imputation:**
  - Categorical columns like `Gender`, `Married`, `Dependents`, `Self_Employed`, and `Loan_Amount_Term` are filled using the mode.
  - Numerical columns like `LoanAmount` are filled using the median.
  - `Credit_History` is filled using the mode due to binary classification nature.

- **Encoding:**
  - Label Encoding for binary categorical features like `Gender`, `Married`, `Education`, and `Self_Employed`.
  - `Dependents` column converted to integer (replacing '3+' with `3`).
  - One-Hot Encoding for `Property_Area` with `drop_first=True` to avoid dummy variable trap.

### 3. Feature Engineering
- Create additional features to enhance predictive power:
  - **Total_Income** = `ApplicantIncome` + `CoapplicantIncome`
  - **EMI** = `LoanAmount` / `Loan_Amount_Term`
  - **Balance_Income** = `Total_Income` - (`EMI` × 1000)

### 4. Exploratory Data Analysis (EDA)
- Visual inspection using `.head()`, `.info()`, `.isna().sum()`, and `.shape()`.
- Plot correlation heatmap for numerical features to understand interdependencies.
- Identify influential features through visual analytics.

### 5. Feature-Target Split
- Separate data into features (`X`) and target (`y`) for training.
- Drop `Loan_ID` and `Loan_Status` from features.
- Create final test features by removing `Loan_ID` and `Loan_Status` from `test.csv`.

### 6. Train-Test Split
- Use `train_test_split` with 80% training and 20% validation data.
- Stratify on `y` to maintain class distribution.

### 7. Pipeline Setup
- Use `ColumnTransformer` with `RobustScaler` to scale numerical features.
- Define pipelines for:
  - **Logistic Regression**
  - **Random Forest** (with hyperparameter tuning using `GridSearchCV`)

### 8. Model Training & Hyperparameter Tuning
- Perform `GridSearchCV` on Random Forest with:
  - `n_estimators`: [50, 100]
  - `max_depth`: [4, 6, None]
  - `min_samples_split`: [2, 5]
- 5-fold cross-validation with `StratifiedKFold`.

### 9. Model Evaluation
- Evaluate both models using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **Confusion Matrix (with Seaborn heatmap)**
- Identify top 10 important features from the Random Forest model and visualize them.

### 10. Model Comparison
- Summarize and compare both models’ performance in a dataframe.
- Choose the best-performing model for final prediction.

### 11. Submission Generation
- Use the best model to predict `Loan_Status` on test data.
- Map predicted labels back to `Y` and `N`.
- Save results to `result.csv` in the required submission format.


## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Output

- `result.csv`: Contains predictions in the format:
