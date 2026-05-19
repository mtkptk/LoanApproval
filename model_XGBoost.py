import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score,  ConfusionMatrixDisplay, precision_recall_curve
)

xg_param_undersampled = {'subsample': 0.8, 'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 1, 'colsample_bytree': 0.8}
xg_param_oversampled = {'subsample': 0.8, 'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 1, 'colsample_bytree': 0.8}


def perform_xgboost(undersample = True):
    DESCRIPTOR = "Undersampled" if undersample else "Oversampled"
    INPUT_PATH = f'./Data/Prepared Data/{DESCRIPTOR}/'
    OUTPUT_PATH = './Outputs/XGB/'

    X_train = pd.read_csv(f'{INPUT_PATH}X_train.csv')
    y_train = pd.read_csv(f'{INPUT_PATH}y_train.csv').squeeze()  # Convert to Series
    X_test  = pd.read_csv(f'{INPUT_PATH}X_test.csv')
    y_test  = pd.read_csv(f'{INPUT_PATH}y_test.csv').squeeze()   # Convert to Series

    print("Data loaded successfully. Starting model training...")

    # Train XGBoost Classifier
    xgb_params = xg_param_undersampled if undersample else xg_param_oversampled
    xgb = XGBClassifier(
        **xgb_params,
        eval_metric='logloss',
        random_state=42
    )

    xgb.fit(X_train, y_train)

    print("Model trained successfully. Evaluating model...")

    # Predictions
    y_pred = xgb.predict(X_test)
    y_pred_proba = xgb.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

    plt.plot(thresholds, precision[:-1], label='Preciznost')
    plt.plot(thresholds, recall[:-1], label='Odziv')
    plt.xlabel('Granica')
    plt.legend()
    plt.show()

    if undersample:
        threshold = 0.729
    else:
        threshold = 0.426

    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Odbijen', 'Odobren'])
    disp.plot(cmap='Blues')
    plt.title('Matrica konfuzije - XGBoost')
    plt.xlabel('Predviđeni ishod')
    plt.ylabel('Stvarni ishod')
    plt.savefig(f'{OUTPUT_PATH}confusion_matrix_{DESCRIPTOR}.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print("=" * 50)
    print("Model Metrics")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': [col for col in X_train.columns],
        'Importance': xgb.feature_importances_
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

perform_xgboost(undersample=True)
perform_xgboost(undersample=False)