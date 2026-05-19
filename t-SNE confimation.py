import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

X_train = pd.read_csv('./Data/Prepared Data/Oversampled/X_train.csv')
X_test = pd.read_csv('./Data/Prepared Data/Oversampled/X_test.csv')
y_train = pd.read_csv('./Data/Prepared Data/Oversampled/y_train.csv').squeeze()
y_test = pd.read_csv('./Data/Prepared Data/Oversampled/y_test.csv').squeeze()

X = pd.concat([X_train, X_test], ignore_index=True)
y = pd.concat([y_train, y_test], ignore_index=True)

# Undersample to 0.05
df_temp = pd.concat([X, y.reset_index(drop=True)], axis=1)
df_temp = df_temp.groupby(y.name, group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=42))
indices = df_temp.index
X = df_temp.drop(columns=[y.name]).reset_index(drop=True)
y = df_temp[y.name].reset_index(drop=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("X je skaliran, pokrećem TSNE...")
tsne = TSNE(n_components=2, random_state=42, )
X_tsne = tsne.fit_transform(X_scaled)
print("TSNE transformacija gotova, dodajem rezultate u DataFrame...")
df = pd.DataFrame()
df['TSNE_1'] = X_tsne[:, 0]
df['TSNE_2'] = X_tsne[:, 1]

plt.figure(figsize=(10, 8))
colors = ['blue' if x == 0 else 'red' for x in y]
plt.scatter(df['TSNE_1'], df['TSNE_2'], c=colors, alpha=0.6, s=20)
plt.xlabel('TSNE_1')
plt.ylabel('TSNE_2')
plt.title('TSNE Visualization (Red: default_ind=1, Blue: default_ind=0)')
plt.legend(['No Default (0)', 'Default (1)'], loc='best')
plt.show()

print("Done")