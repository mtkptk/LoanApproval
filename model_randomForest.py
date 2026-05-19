import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay,  precision_recall_curve

)

rf_undersampled_params = {
    'n_estimators': 300,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'max_depth': None,
    'n_jobs': -1,
    'random_state': 42
}

rf_oversampled_params = {
    'n_estimators': 200,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'log2',
    'max_depth': None,
    'n_jobs': -1,
    'random_state': 42
}

def perform_random_forest(undersample = True):
    DESCRIPTOR = "Undersampled" if undersample else "Oversampled"
    INPUT_PATH = f'./Data/Prepared Data/{DESCRIPTOR}/'
    OUTPUT_PATH = './Outputs/RF/'

    X_train = pd.read_csv(f'{INPUT_PATH}X_train.csv')
    y_train = pd.read_csv(f'{INPUT_PATH}y_train.csv').squeeze()
    X_test  = pd.read_csv(f'{INPUT_PATH}X_test.csv')
    y_test  = pd.read_csv(f'{INPUT_PATH}y_test.csv').squeeze()

    print("Data loaded successfully. Starting model training...")

    rf_params = rf_undersampled_params if undersample else rf_oversampled_params
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

    plt.plot(thresholds, precision[:-1], label='Preciznost')
    plt.plot(thresholds, recall[:-1], label='Odziv')
    plt.xlabel('Granica')
    plt.legend()
    plt.show()

    if undersample:
        threshold = 0.642
    else:
        threshold = 0.462

    y_pred = (y_pred_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Odbijen', 'Odobren'])
    disp.plot(ax = ax, cmap='Blues')
    plt.title('Matrica konfuzije - Random Forest')
    plt.tight_layout()
    plt.xlabel('Predviđeni ishod')
    plt.ylabel('Stvarni ishod')
    plt.savefig(f'{OUTPUT_PATH}confusion_matrix_{DESCRIPTOR}.png', dpi=300, bbox_inches='tight')
    plt.clf()

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print("=" * 50)
    print("Model Metrics")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    feature_importance = pd.DataFrame({
        'Feature': [col for col in X_train.columns],
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.barh(feature_importance['Feature'][:20], feature_importance['Importance'][:20])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}feature_importance_{DESCRIPTOR}.png', dpi=300, bbox_inches='tight')
    plt.clf()

    print("\nTop 10 Features:")
    print(feature_importance.head(10))
    
    return

perform_random_forest(undersample=True)
perform_random_forest(undersample=False)