import pandas as pd
import os
import logging

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at: {path}")
    df = pd.read_csv(path)
    logging.info(f"Loaded dataset with shape: {df.shape}")
    return df

def check_data_integrity(df: pd.DataFrame) -> None:
    logging.info("Checking data integrity...")
    logging.info(df.info())
    logging.info(df.isnull().sum())
    logging.info(df.nunique())