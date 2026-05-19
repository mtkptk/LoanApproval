import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

df = pd.read_csv('./Data/loan_data.csv')

# Zanemariti kreditiranje starih ljudi i ljudi sa primanjima od 1_000_000
df = df[df['person_age'] <= 70]
df = df[df['person_income'] <= 1_000_000]

numeric_df = df.select_dtypes(include=[np.number])

fig, axes = plt.subplots(len(numeric_df.columns), 1, figsize=(10, 4*len(numeric_df.columns)))
if len(numeric_df.columns) == 1:
    axes = [axes]
for idx, col in enumerate(numeric_df.columns):
    axes[idx].hist(numeric_df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribucija kolone {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frekvencija')
    axes[idx].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{float(x):n}'))
    axes[idx].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{float(x):n}'))
    axes[idx].spines['top'].set_visible(False)
    axes[idx].spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('./Outputs/Descriptions/univariate_distributions.png', dpi=300)
plt.close()
