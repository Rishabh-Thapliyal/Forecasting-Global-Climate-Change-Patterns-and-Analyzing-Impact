import os
import re
import glob
import pandas as pd
'''
This file can get all cities and global temperature at every year's every month
Add global and each city baseline temperature to baseline_temps
Such as baseline[cityidx][monthidx-1] is cityidx's monthidx baseline temperature, baseline[0][monthidx-1] is global monthidx baseline temperature
Store anomaly in ./Data/Temperature which contains global and all city data(format:.txt)
Output .csv result to default path, global and each city has a .csv result
'''
# Baseline temperatures for global and 20 cities(from global to city1, then from city1-20; from Jan to Dec)
baseline_temps = [
    [12.58, 12.82, 13.445, 14.32, 15.21, 15.87, 16.18, 16.06, 15.47, 14.525, 13.515, 12.805],
    [-4.87, -2.10, 4.85, 12.98, 19.85, 24.24, 26.29, 25.08, 20.00, 12.88, 4.45, -2.77],
    [14.37, 17.70, 23.53, 29.45, 33.53, 34.18, 30.98, 29.66, 29.28, 26.28, 20.64, 15.61],
    [13.94, 16.59, 21.49, 25.15, 30.33, 33.97, 35.31, 34.74, 31.48, 25.76, 20.57, 15.06],
    [-10.34, -9.21, -3.97, 4.90, 12.01, 16.21, 17.81, 16.30, 10.84, 4.48, -1.78, -6.59],
    [10.18, 10.77, 11.84, 14.80, 19.22, 23.47, 25.64, 25.61, 22.60, 18.52, 15.40, 11.95],
    [0.83, 1.73, 5.16, 11.11, 16.03, 19.74, 23.57, 24.76, 20.46, 14.17, 8.80, 3.57],
    [3.06, 3.50, 5.65, 8.06, 11.64, 14.76, 16.47, 16.22, 13.95, 10.37, 6.19, 4.10],
    [-4.66, -3.69, -2.84, -0.40, 3.52, 6.76, 8.46, 7.71, 4.87, 1.05, -2.40, -4.46],
    [-11.04, -7.96, -4.53, -0.48, 3.88, 7.48, 7.73, 6.80, 5.00, 0.01, -5.80, -10.11],
    [23.18, 22.88, 21.61, 18.49, 14.96, 11.69, 11.86, 14.86, 19.04, 21.54, 22.15, 22.66],
    [9.13, 11.13, 13.69, 15.69, 19.71, 23.10, 27.00, 26.83, 23.26, 18.50, 13.51, 9.2],
    [-17.22, -13.15, -9.40, -2.27, 4.31, 9.58, 11.65, 9.90, 4.94, -3.32, -10.96, -16.84],
    [13.31, 13.48, 13.32, 14.06, 15.39, 16.95, 19.23, 20.04, 20.18, 18.50, 16.19, 14.03],
    [-3.10, -2.00, 3.05, 9.39, 15.07, 19.99, 22.60, 21.70, 17.63, 11.37, 5.65, -0.73],
    [2.82, 3.84, 6.49, 9.15, 13.02, 16.17, 18.08, 17.81, 15.36, 11.05, 6.41, 3.83],
    [3.07, 4.35, 6.59, 9.50, 14.02, 18.19, 20.94, 20.73, 17.30, 12.32, 7.93, 4.30],
    [10.18, 10.46, 11.70, 13.34, 16.91, 20.13, 22.82, 23.16, 21.45, 17.58, 13.52, 11.01],
    [-6.25, -7.35, -3.94, 1.75, 8.05, 13.64, 15.90, 15.25, 10.68, 5.78, 1.28, -2.92],
    [19.54, 19.84, 19.74, 18.20, 16.21, 14.74, 13.66, 14.06, 15.03, 16.25, 17.70, 18.91],
    [22.97, 23.14, 23.19, 22.60, 21.64, 20.50, 20.13, 21.88, 23.72, 23.63, 23.05, 22.63],
]

# folder which contain all data
folder_path = "./Data/Temperature"  

def fetch_temperature_table(folder_path, baseline_temps):
    # Get all .txt files in the folder, sorted by filename's first integer number i.e. from 0-20, 0 is global, 1-20 is cityidx
    file_list = sorted(glob.glob(os.path.join(folder_path, "*.txt")), key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

    # Ensure the number of files equals to the baseline temperature list
    if len(file_list) != len(baseline_temps):
        raise ValueError("Number of files does not match the baseline temperature list, please check your data.")

    # Read and process each file
    for idx, file_path in enumerate(file_list):
        city_baseline = baseline_temps[idx]  # Get the baseline temperatures for the corresponding city
        data = []
        
        # Try opening the file with different encoding (ISO-8859-1 or Latin-1) to avoid UnicodeDecodeError
        try:
            with open(file_path, "r", encoding='utf-8') as file:
                for line in file:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            year = int(parts[0])
                            month = int(parts[1])
                            
                            # Validate month value (1-12)
                            if 1 <= month <= 12:
                                anomaly = float(parts[2]) if parts[2] != 'NaN' else float('nan')
                                data.append((year, month, anomaly))
                        except ValueError:
                            continue  # Skip invalid lines
        except UnicodeDecodeError:
            with open(file_path, "r", encoding='ISO-8859-1') as file:
                for line in file:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            year = int(parts[0])
                            month = int(parts[1])
                            
                            # Validate month value (1-12)
                            if 1 <= month <= 12:
                                anomaly = float(parts[2]) if parts[2] != 'NaN' else float('nan')
                                data.append((year, month, anomaly))
                        except ValueError:
                            continue  # Skip invalid lines

        # Calculate the actual temperatures
        results = []
        for year, month, anomaly in data:
            if pd.notna(anomaly):
                baseline_temp = city_baseline[month - 1]
                actual_temp = baseline_temp + anomaly
                results.append((year, month, actual_temp))
            else:
                results.append((year, month, float('nan')))

        # Save as a .csv file for every input
        city_name = os.path.basename(file_path).replace(".txt", "")
        df = pd.DataFrame(results, columns=["Year", "Month", "Actual Temperature"])
        df["Actual Temperature"] = df["Actual Temperature"].round(3)  # Round to 3 decimal places
        df.to_csv(f"{city_name}_actual_temperatures.csv", index=False)
        print(f"Actual temperatures for {city_name} calculated and saved as {city_name}_actual_temperatures.csv.")

fetch_temperature_table(folder_path, baseline_temps)