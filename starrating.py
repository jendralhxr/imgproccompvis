# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 11:25:04 2026

@author: rdx
"""

import pandas as pd

# Load the dataset
file_path = 'unisza-dip.csv'
df = pd.read_csv(file_path)

# 1. Count how many unique email addresses (Column B)
unique_email_count = df['Email Address'].nunique()
print(f"Number of unique email addresses: {unique_email_count}")

# 2. For each combination of Email Address and Evaluee, retain only the last timestamp
# Ensure Timestamp is in datetime format for accurate sorting
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Sort by Timestamp to guarantee 'last' is the most recent
df = df.sort_values('Timestamp')

# Drop duplicates based on Email and the Group being evaluated, keeping the last occurrence
df_cleaned = df.drop_duplicates(subset=['Email Address', 'The group being evaluated'], keep='last')

# Display the first few rows of the cleaned data
#print(df_cleaned.head())

raw_counts = df['Email Address'].value_counts()
cleaned_counts = df_cleaned['Email Address'].value_counts()

# specific check: combine them into a dataframe for easier viewing
comparison = pd.DataFrame({'Raw Submissions': raw_counts, 'Unique Groups Evaluated': cleaned_counts})
comparison = comparison.fillna(0).astype(int)
comparison = comparison.sort_values('Unique Groups Evaluated', ascending=False)

print("Counts per Email Address (sorted by Unique Groups Evaluated):")
print(comparison.to_markdown(numalign="left", stralign="left"))


# no one is perfect, I guess
condition_777 = (
    (df.iloc[:, 3] == 7) & 
    (df.iloc[:, 4] == 7) & 
    (df.iloc[:, 5] == 7)
)

# Keep rows where the condition is NOT met (~)
df_filtered = df[~condition_777]

# Verify the result
#print(f"Original clean count: {len(df)}")
#print(f"Count after dropping 7-7-7: {len(df_filtered)}")
#print(df_filtered.head())

df_filtered

average_ratings = df_filtered.groupby('The group being evaluated')[rating_columns].mean()

# Optional: Add an Overall Score (average of the 3 components)
average_ratings['Overall Score'] = average_ratings.mean(axis=1)

# Sort by Overall Score for better readability
average_ratings = average_ratings.sort_values('Overall Score', ascending=False)

col_clarity = 'How easy to understand (clarity of the logical construct)'
col_reliability = 'The program works as intended (accuracy, reliability, reproducibility)'
col_applicability = 'How [potentially] applicable/useful/adaptable in real life (niche/limited scope is also okay)'
separate_averages = df_filtered.groupby('The group being evaluated')[[col_clarity, col_reliability, col_applicability]].mean()
separate_averages.columns = ['Avg_Clarity', 'Avg_Reliability', 'Avg_Applicability']



