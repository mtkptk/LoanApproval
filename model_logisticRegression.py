import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay, precision_recall_curve
)

lr_param_undersampled = {'solver': 'lbfgs', 'penalty': 'l2', 'max_iter': 500, 'C': 0.1}
lr_param_oversampled = {'solver': 'liblinear', 'penalty': 'l2', 'max_iter': 500, 'C': 100}

def perform_logistic_regression(undersample = True):
    
    DESCRIPTOR = "Undersampled" if undersample else "Oversampled"
    INPUT_PATH = f'./Data/Prepared Data/{DESCRIPTOR}/'
    OUTPUT_PATH = './Outputs/LR/'

    X_train = pd.read_csv(f'{INPUT_PATH}X_train.csv')
    y_train = pd.read_csv(f'{INPUT_PATH}y_train.csv').squeeze()  # Convert to Series
    X_test  = pd.read_csv(f'{INPUT_PATH}X_test.csv')
    y_test  = pd.read_csv(f'{INPUT_PATH}y_test.csv').squeeze()   # Convert to Series
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr_params = lr_param_undersampled if undersample else lr_param_oversampled
    lr = LogisticRegression(
        **lr_params,
        random_state=42
    )
    lr.fit(X_train_sc, y_train)

    y_pred = lr.predict(X_test_sc)
    y_pred_proba = lr.predict_proba(X_test_sc)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

    plt.plot(thresholds, precision[:-1], label='Preciznost')
    plt.plot(thresholds, recall[:-1], label='Odziv')
    plt.xlabel('Granica')
    plt.legend()
    plt.show()

    if undersample:
        threshold = 0.746
    else:
        threshold = 0.482

    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("=" * 50)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 50)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Odbijen", "Odobren"])
    disp.plot(ax=ax, cmap='Blues')
    plt.title("Matrica konfuzije - Logistic Regression")
    plt.xlabel('Predviđeni ishod')
    plt.ylabel('Stvarni ishod')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}confusion_matrix_{DESCRIPTOR}.png', dpi=300, bbox_inches='tight')
    plt.clf()

    feature_names = X_train.columns
    coefficients = lr.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feature_importance['Feature'][:15], feature_importance['Coefficient'][:15])
    plt.xlabel('Coefficient Value')
    plt.title('Top 15 Feature Importance - Logistic Regression')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}feature_importance_{DESCRIPTOR}.png', dpi=300, bbox_inches='tight')
    plt.close()

    feature_importance.to_csv(f'{OUTPUT_PATH}feature_importance_{DESCRIPTOR}.csv', index=False)

    return

perform_logistic_regression(undersample=True)
perform_logistic_regression(undersample=False)
