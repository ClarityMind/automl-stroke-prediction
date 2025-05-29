# AutoML Pipeline for Stroke Prediction

## Project Overview
This project implements an automated machine learning (AutoML) pipeline designed for stroke prediction using clinical data. It covers the entire workflow from data loading and preprocessing to model training, hyperparameter optimization, and report generation.

The pipeline supports multiple classification models — Logistic Regression, Random Forest, Support Vector Machine (SVM), and XGBoost — and uses Optuna for efficient hyperparameter tuning. Data profiling reports are generated using `ydata-profiling`, while experiment tracking and logging are handled with `mlflow`.

---

## Dataset
The project uses the **Stroke Prediction Dataset**(https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle, which includes various clinical features such as age, gender, BMI, average glucose level, smoking status, and others. The target variable is binary, indicating whether a stroke has occurred.

---

## Features

- **Data Loading**: Reads raw CSV data and checks integrity.
- **Preprocessing**: Handles missing values, encodes categorical variables, scales features, and optionally applies PCA for dimensionality reduction.
- **Model Training**: Implements Logistic Regression, Random Forest, SVM, and XGBoost classifiers.
- **Hyperparameter Optimization**: Utilizes Optuna to find optimal model parameters automatically.
- **Report Generation**: Produces detailed HTML reports on data profiling and model evaluation using `ydata-profiling` and logs experiments via `mlflow`.
- **Configurable Pipeline**: Easily adjustable through `config.yaml` for paths, model selection, preprocessing options, and optimization settings.

---

## Project Structure

```
automl-stroke-prediction/
│
├── config/
│ └── config.yaml # Configuration file for pipeline parameters
│
├── data/
│ ├── raw/ # Raw input data files (stroke.csv)
│ └── processed/ # Processed datasets (after preprocessing)
│
├── reports/ # Generated profiling and model reports
│
├── src/
│ ├── data_loader.py # Data loading and integrity checking
│ ├── preprocessing.py # Data cleaning and feature engineering
│ ├── model_selection.py # Model definitions and evaluation
│ ├── optimization.py # Hyperparameter tuning with Optuna
│ ├── report_generator.py # Generation of data and model reports
│ └── pipeline.py # Orchestrates the entire AutoML pipeline
│
├── notebooks/
│ └── exploratory.ipynb # Initial data analysis and visualization
│
├── run_pipeline.py # Script to run the full pipeline
├── requirements.txt # Project dependencies
└── README.md # Project documentation (this file)
```
---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/automl-stroke-prediction.git
cd automl-stroke-prediction
```
2. (Optional) Create and activate a virtual environment.

3. Install dependencies:

```bash
pip install -r requirements.txt
```
4. Download the Stroke Prediction Dataset from Kaggle and place stroke.csv in the data/raw/ folder.

## Usage
Run the full pipeline using the command:

```bash
python run_pipeline.py
```
The pipeline will:

* Load and preprocess the data.

* Train and evaluate the selected models.

* Optimize hyperparameters with Optuna.

* Generate detailed profiling reports.

* Log experiments via MLflow.

## Configuration
Modify ```config/config.yaml``` to customize:

* Data path

* Models to use

* Whether to apply PCA

* Hyperparameter optimization method

* Reporting options

## Results
Generated reports and logs are saved in the reports/ directory, including:

* HTML data profiling reports (ydata-profiling)

* MLflow experiment tracking with model metrics and hyperparameters

* Visualizations of model performance

## Dependencies
* Python 3.8+

* pandas

* scikit-learn

* xgboost

* optuna

* mlflow

* ydata-profiling (formerly pandas-profiling)