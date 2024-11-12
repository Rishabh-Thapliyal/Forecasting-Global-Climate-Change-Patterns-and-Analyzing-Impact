# First, filter the necessary columns and perform the required calculations
# Rename columns to lowercase for consistency
import pandas as pd

file_path = 'volcano-events.tsv'
volcano_data = pd.read_csv(file_path, sep='\t')


# Extract relevant columns
volcano_data_relevant = volcano_data[['Year', 'VEI']].copy()

# Drop rows with missing values in 'Year' or 'VEI'
volcano_data_relevant.dropna(subset=['Year', 'VEI'], inplace=True)

# Convert 'Year' to integer for grouping purposes
volcano_data_relevant['Year'] = volcano_data_relevant['Year'].astype(int)

# Calculate the yearly average VEI and number of events per year
yearly_stats = volcano_data_relevant.groupby('Year').agg(
    year_average_VEI=('VEI', 'mean'),
    year_events_amount=('VEI', 'count')
).reset_index()

# Merge the calculated stats back with the original relevant data to include each row with the yearly stats
result_df = volcano_data_relevant.merge(yearly_stats, on='Year')
result_df.rename(columns={'Year': 'year', 'VEI': 'VEI'}, inplace=True)

# Save the result to a new CSV file
output_path = 'volcano_yearly_VEI_stats.csv'
result_df.to_csv(output_path, index=False)

output_path