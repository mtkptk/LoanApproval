import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def create_features(df):
    
    df = df[df['person_age'] <= 70]
    df = df[df['person_income'] <= 1_000_000]

    df = pd.get_dummies(df, columns=['person_gender'], drop_first=False)
    df = pd.get_dummies(df, columns=['person_education'], drop_first=False)
    df = pd.get_dummies(df, columns=['person_home_ownership'], drop_first=False)
    df = pd.get_dummies(df, columns=['loan_intent'], drop_first=False)
    df = pd.get_dummies(df, columns=['previous_loan_defaults_on_file'], drop_first=True)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    target = 'loan_status'
    features = [col for col in df.columns if col != target]

    return df, features, target

def undersample_data(X, y, reference_class = 1):
    reference_indices = y[y == reference_class].index
    other_indices = y[y != reference_class].index
    undersampled_other_indices = other_indices.to_series().sample(n=len(reference_indices), random_state=42).index
    undersampled_indices = reference_indices.union(undersampled_other_indices)
    return X.loc[undersampled_indices], y.loc[undersampled_indices]

def oversample_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
    return X_train_oversampled, y_train_oversampled

df = pd.read_csv('./Data/loan_data.csv')

df, features, target = create_features(df)

df = df[features + [target]]

df.describe(include='all').T.to_excel('./Outputs/Descriptions/feature_summary.xlsx')

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_undersampled, y_train_undersampled = undersample_data(X_train, y_train, reference_class=1)

OUTPUT_PATH = './Data/Prepared Data/Undersampled/'

X_train_undersampled.to_csv(f'{OUTPUT_PATH}X_train.csv', index=False)
y_train_undersampled.to_csv(f'{OUTPUT_PATH}y_train.csv', index=False)
X_test.to_csv(f'{OUTPUT_PATH}X_test.csv', index=False)
y_test.to_csv(f'{OUTPUT_PATH}y_test.csv', index=False)


X_train_oversampled, y_train_oversampled = oversample_data(X_train, y_train)

OUTPUT_PATH = './Data/Prepared Data/Oversampled/'

X_train_oversampled.to_csv(f'{OUTPUT_PATH}X_train.csv', index=False)
y_train_oversampled.to_csv(f'{OUTPUT_PATH}y_train.csv', index=False)
X_test.to_csv(f'{OUTPUT_PATH}X_test.csv', index=False)
y_test.to_csv(f'{OUTPUT_PATH}y_test.csv', index=False)

print("Features created.")