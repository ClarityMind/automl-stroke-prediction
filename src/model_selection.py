from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_models():
    return {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'SVM': SVC(class_weight='balanced', probability=True, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

def evaluate_model(model, X, y, cv: int = 5) -> dict:
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
    scores = {}
    for metric in scoring_metrics:
        scores[metric] = cross_val_score(model, X, y, cv=cv, scoring=metric).mean()
    return scores

def train_and_evaluate(models: dict, X, y) -> dict:
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X, y)
    return results
