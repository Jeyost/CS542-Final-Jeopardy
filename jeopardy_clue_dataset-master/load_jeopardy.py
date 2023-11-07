import pandas as pd

df = pd.read_csv('combined_season1-39.tsv', sep='\t')
print(df['answer'])
print(df['question'])

s1_df = pd.read_csv('seasons/season1.tsv', sep='\t')