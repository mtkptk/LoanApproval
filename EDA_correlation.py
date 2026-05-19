import locale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

df = pd.read_csv('./Data/loan_data.csv')

# Zanemariti kreditiranje starih ljudi i ljudi sa primanjima od 1_000_000
df = df[df['person_age'] <= 70]
df = df[df['person_income'] <= 1_000_000]

numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
correlation_matrix.to_excel("./Outputs/Descriptions/correlation_matrix.xlsx")

# Heatmap
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', cbar_kws={'format': plt.FuncFormatter(lambda x, _: f'{x:.2f}'.replace('.', ','))}, mask=mask)
ax = plt.gca()

for text in ax.collections[0].axes.texts:
    text.set_text(text.get_text().replace('.', ','))
    
plt.tight_layout()
plt.savefig('./Outputs/Descriptions/correlation_heatmap.png', dpi=300)
plt.close()

