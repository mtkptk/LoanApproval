import pandas as pd

df = pd.read_csv('./Data/loan_data.csv')

df.describe(include='all').T.to_excel("./Outputs/Descriptions/inital_dataset_description.xlsx")
