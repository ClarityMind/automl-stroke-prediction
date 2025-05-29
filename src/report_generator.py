import pandas as pd
from ydata_profiling import ProfileReport
import mlflow
import os
import json

def generate_data_profile(df: pd.DataFrame, output_path: str):
    profile = ProfileReport(df, title="Data Report", explorative=True)
    profile.to_file(output_path)

def log_model_metrics(model_name: str, metrics: dict, params: dict):
    mlflow.start_run(run_name=model_name)
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    for p, v in params.items():
        mlflow.log_param(p, v)
    mlflow.end_run()

def save_metrics_to_file(results: dict, output_path: str = "reports/metrics_summary.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)