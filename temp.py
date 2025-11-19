import pandas as pd
import os

# Set the path to the directory where your CSV files are stored
folder_path = 'models/test1'

# List all CSV files in the directory
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Read each CSV file and store them in a list
dataframes = []
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, nrows=950)
    dataframes.append(df)
# Concatenate all DataFrames into one
merged_dataframe = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_dataframe.to_csv('merged_output.csv', index=False)

print("CSV files have been merged and saved as 'merged_output.csv'.")
