import logging
from src.pipeline import run_pipeline

def main():
    logging.basicConfig(level=logging.INFO)
    try:
        run_pipeline("config/config.yaml")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()