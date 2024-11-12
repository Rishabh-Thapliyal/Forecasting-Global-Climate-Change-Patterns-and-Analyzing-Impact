import pandas as pd

# Load the data, skipping the header rows
data = pd.read_csv("CO2Release.txt", delim_whitespace=True, comment='#',
                   names=["year", "month", "decimal_date", "monthly_average", "deseasonalized", "days", "st_dev", "uncertainty"])

# Select only the first four columns, excluding 'decimal_date'
data_subset = data[["year", "month", "monthly_average"]]

# Save the result to a new CSV file
data_subset.to_csv("CO2.csv", index=False)

print("CSV file created: output.csv")