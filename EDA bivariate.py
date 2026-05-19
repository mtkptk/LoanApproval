from matplotlib import ticker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import locale

locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

df = pd.read_csv('./Data/loan_data.csv')

# Zanemariti kreditiranje starih ljudi i ljudi sa primanjima od 1_000_000
df = df[df['person_age'] <= 70]
df = df[df['person_income'] <= 1_000_000]

numeric_df = df.select_dtypes(include=[np.number])

plt.rcParams['axes.formatter.use_locale'] = True

numeric_cols = numeric_df.columns.tolist()

if len(numeric_cols) > 1:
    for i in range(len(numeric_cols)):
        y_cols = [numeric_cols[j] for j in range(len(numeric_cols)) if i != j]
        num_plots = len(y_cols)
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5*min(5, num_plots)))
        if num_plots == 1:
            axes = [axes]
        
        for plot_idx, y_col in enumerate(y_cols):
            axes[plot_idx].scatter(numeric_df[numeric_cols[i]], numeric_df[y_col], alpha=0.5)
            axes[plot_idx].set_xlabel(numeric_cols[i])
            axes[plot_idx].set_ylabel(y_col)
            axes[plot_idx].set_title(f'Scatter: {numeric_cols[i]} vs {y_col}')
            axes[plot_idx].ticklabel_format(style='plain', axis='both')
            axes[plot_idx].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):n}'))
            axes[plot_idx].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):n}'))
        
        plt.tight_layout()
        plt.savefig(f'./Outputs/Descriptions/bivariate_scatters/{numeric_cols[i]}.png', dpi=300)
        plt.close()

