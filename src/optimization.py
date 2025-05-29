import optuna
from sklearn.model_selection import cross_val_score
from src.model_selection import get_models

def optimize_model(model_name: str, X, y, method: str = 'optuna'):
    if method != 'optuna':
        raise ValueError("Only 'optuna' is supported in this version.")

    def objective(trial):
        models = get_models()
        if model_name == 'LogisticRegression':
            C = trial.suggest_loguniform('C', 0.01, 10)
            model = models[model_name].set_params(C=C)
        elif model_name == 'RandomForest':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            model = models[model_name].set_params(n_estimators=n_estimators, max_depth=max_depth)
        elif model_name == 'SVM':
            C = trial.suggest_loguniform('C', 0.1, 10)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
            model = models[model_name].set_params(C=C, kernel=kernel)
        elif model_name == 'XGBoost':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
            model = models[model_name].set_params(n_estimators=n_estimators, learning_rate=learning_rate)
        else:
            raise ValueError("Unsupported model")

        score = cross_val_score(model, X, y, cv=3, scoring='f1').mean()

        print(f"Trial {trial.number} - Model: {model_name}, F1: {score:.4f}, Params: {trial.params}")
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print(f"Best trial for {model_name}: Value: {study.best_value:.4f}, Params: {study.best_params}")
    return study.best_params, study.best_value
