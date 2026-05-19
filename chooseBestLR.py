import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, precision_recall_curve, auc
from pathlib import Path
from sklearn.preprocessing import StandardScaler

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 200, 500],
}

def train_and_evaluate_lr(X_train, y_train, X_test, y_test, filename="Model"):

    lr_grid = RandomizedSearchCV(
        LogisticRegression(random_state=42),
        lr_param_grid,
        cv=skf,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        n_iter=20,
        random_state=42
    )
    
    lr_grid.fit(X_train, y_train)
    
    output_dir = Path(f'./Outputs/LR/Strategies/')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f'{filename}.txt', 'w') as f:
        f.write(f"Best parameters: {lr_grid.best_params_}\n")
        f.write(f"Best CV AUC-ROC score: {lr_grid.best_score_:.4f}\n")
    

    best_lr = lr_grid.best_estimator_

    y_pred = best_lr.predict(X_test)
    y_pred_proba = best_lr.predict_proba(X_test)[:, 1]
    
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    # Write metrics to output file
    with open(output_dir / f'{filename}.txt', 'a') as f:
        f.write("\n=== Test Set Performance ===\n")
        f.write(f"AUC-ROC: {auc_roc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
    
    return best_lr, {
        'auc_roc': auc_roc,
        'f1': f1,
        'pr_auc': pr_auc,
        'model': best_lr
    }

X_train = pd.read_csv('./Data/Prepared Data/Undersampled/X_train.csv')
y_train = pd.read_csv('./Data/Prepared Data/Undersampled/y_train.csv').squeeze()
X_test = pd.read_csv('./Data/Prepared Data/Undersampled/X_test.csv')
y_test = pd.read_csv('./Data/Prepared Data/Undersampled/y_test.csv').squeeze()

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

best_lr_model, lr_metrics = train_and_evaluate_lr(X_train_sc, y_train, X_test_sc, y_test, filename="Undersampled")

X_train = pd.read_csv('./Data/Prepared Data/Oversampled/X_train.csv')
y_train = pd.read_csv('./Data/Prepared Data/Oversampled/y_train.csv').squeeze()
X_test = pd.read_csv('./Data/Prepared Data/Oversampled/X_test.csv')
y_test = pd.read_csv('./Data/Prepared Data/Oversampled/y_test.csv').squeeze()

X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

best_rf_model, rf_metrics = train_and_evaluate_lr(X_train_sc, y_train, X_test_sc, y_test, filename="Oversampled")

