import pandas as pd

data = pd.read_csv('population_raw.csv')

# Select the required columns and rename them
data_subset = data[['Variable observation date', 'Variable observation value']].rename(
    columns={'Variable observation date': 'year', 'Variable observation value': 'population'}
)

# Save the modified data to a new CSV file
output_path = 'population_modified.csv'
data_subset.to_csv(output_path, index=False)

output_path