import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, precision_recall_curve, auc
from pathlib import Path

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

def train_and_evaluate_rf(X_train, y_train, X_test, y_test, filename="Model"):

    rf_grid = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        rf_param_grid,
        cv=skf,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        n_iter=20,
        random_state=42
    )
    
    rf_grid.fit(X_train, y_train)
    
    output_dir = Path(f'./Outputs/RF/Strategies/')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f'{filename}.txt', 'w') as f:
        f.write(f"Best parameters: {rf_grid.best_params_}\n")
        f.write(f"Best CV AUC-ROC score: {rf_grid.best_score_:.4f}\n")
    
    best_rf = rf_grid.best_estimator_
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    with open(output_dir / f'{filename}.txt', 'a') as f:
        f.write("\n=== Test Set Performance ===\n")
        f.write(f"AUC-ROC: {auc_roc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
    
    return best_rf, {
        'auc_roc': auc_roc,
        'f1': f1,
        'pr_auc': pr_auc,
        'model': best_rf
    }

X_train = pd.read_csv('./Data/Prepared Data/Undersampled/X_train.csv')
y_train = pd.read_csv('./Data/Prepared Data/Undersampled/y_train.csv').squeeze()
X_test = pd.read_csv('./Data/Prepared Data/Undersampled/X_test.csv')
y_test = pd.read_csv('./Data/Prepared Data/Undersampled/y_test.csv').squeeze()


best_rf_model, rf_metrics = train_and_evaluate_rf(X_train, y_train, X_test, y_test, filename="Undersampled")


X_train = pd.read_csv('./Data/Prepared Data/Oversampled/X_train.csv')
y_train = pd.read_csv('./Data/Prepared Data/Oversampled/y_train.csv').squeeze()
X_test = pd.read_csv('./Data/Prepared Data/Oversampled/X_test.csv')
y_test = pd.read_csv('./Data/Prepared Data/Oversampled/y_test.csv').squeeze()

best_rf_model, rf_metrics = train_and_evaluate_rf(X_train, y_train, X_test, y_test, filename="Oversampled")

