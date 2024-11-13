import pandas as pd
import os
'''
This file contains a function add_location_to_temp(file_path, city_info)
It can add city's information to any existing .csv
tip: I keep the cityidx for every city. -- Harry Liu
'''
city_data = [
    ["0_Global", "Global", "ALL", "ALL"],
    ["1_Peking", "China", "39.38 N", "116.53 E"],
    ["2_NewDelhi", "India", "28.13 N", "77.27 E"],
    ["3_Riyadh", "Saudi Arabia", "24.92 N", "46.11 E"],
    ["4_Moscow", "Russia", "55.45 N", "36.85 E"],
    ["5_Athens", "Greece", "37.78 N", "24.41 E"],
    ["6_Tokyo", "Japan", "36.17 N", "139.23 E"],
    ["7_London", "Britain", "52.24 N", "0.00 W"],
    ["8_Reykjavík", "Iceland", "65.09 N", "21.06 W"],
    ["9_Lhasa", "China", "29.74 N", "90.46 E"],
    ["10_Pretoria", "South Africa", "24.92 S", "28.37 E"],
    ["11_Marrakesh", "Morocco", "31.35 N", "7.54 W"],
    ["12_Anchorage", "USA", "61.88 N", "151.13 W"],
    ["13_SanDiego", "USA", "32.95 N", "117.77 W"],
    ["14_NewYork", "USA", "40.99 N", "74.56 W"],
    ["15_Paris", "France", "49.03 N", "2.45 E"],
    ["16_Rome", "Italy", "42.59 N", "13.09 E"],
    ["17_Barcelona", "Spain", "40.99 N", "2.13 E"],
    ["18_Helsinki", "Finland", "60.27 N", "25.95 E"],
    ["19_Sydney", "Australia", "34.56 S", "151.78 E"],
    ["20_Brasília", "Brazil", "15.27 S", "47.50 W"]
]

# Define a function to add columns to a file
def add_location_to_temp(file_path, city_info):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Unpack city information
    city_name, country, latitude, longitude = city_info
    
    # Check if city name in file path matches the city information
    if city_name in os.path.basename(file_path):
        # Add the new columns
        df['City'] = city_name
        df['Country'] = country
        df['Latitude'] = latitude
        df['Longitude'] = longitude
        
        # Save the modified file
        df.to_csv(file_path, index=False)
        print(f"Processed {file_path} with city {city_name}")
    else:
        print(f"Warning: City name mismatch for file {file_path}")

folder_path = "."
# Process each file
file_paths = [os.path.join(folder_path, f"{city[0]}_actual_temperatures.csv") for city in city_data]
for file_path, city_info in zip(file_paths, city_data):
    add_location_to_temp(file_path, city_info)