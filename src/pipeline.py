import yaml
from src.data_loader import load_data, check_data_integrity
from src.preprocessing import preprocess_pipeline
from src.model_selection import get_models, train_and_evaluate
from src.optimization import optimize_model
from src.report_generator import generate_data_profile, log_model_metrics, save_metrics_to_file

def run_pipeline(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    df = load_data(config['data_path'])
    check_data_integrity(df)

    target = config['target_column']
    y = df[target]
    X = df.drop(columns=[target])

    X = preprocess_pipeline(X, config['use_pca'], config['pca_components'])

    models = get_models()
    all_results = {}

    if config['optimization']:
        for name in config['models']:
            best_params, best_score = optimize_model(name, X, y, method='optuna')
            log_model_metrics(name, {'f1': best_score}, best_params)
            all_results[name] = {'f1': best_score, **best_params}
    else:
        results = train_and_evaluate(models, X, y)
        for name, metrics in results.items():
            log_model_metrics(name, metrics, {})
            all_results[name] = metrics

    save_metrics_to_file(all_results)

    if config['report']:
        generate_data_profile(df, "reports/data_profile.html")
