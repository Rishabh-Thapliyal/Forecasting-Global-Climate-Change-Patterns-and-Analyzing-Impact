'''
after EDA, prepare the data for forecasting
'''

def load_data(start_year, end_year):
    '''
    function to load the data from data_dir_path. 
    Data for all the cities is combined, filtered, and then
    returned as a dataframe.

    :param start_year: year from which you want the data
    :param end_year: year upto which you want the data
    :return: combined dataframe with all the features
    '''

    import pandas as pd
    import glob
    from functools import reduce
    from time import perf_counter

    assert start_year <= end_year
    assert start_year > 0 and end_year > 0
    assert isinstance(start_year, int) and isinstance(end_year, int)

    start_time = perf_counter()

    df_list = []
    data_dir_path = "../../Data/ResultData/"

    # read temperature data for all the cities
    for file_name in glob.glob(data_dir_path + 'temperature/*.csv'):
        if file_name not in [data_dir_path+'temperature/0_Global_air_actual_temperatures.csv', data_dir_path+'temperature/0_Global_water_actual_temperatures.csv']:
            df = pd.read_csv(file_name, low_memory=False)
            df_list.append(df)

    # read global air temperature data and global water temperature data
    df_air_global_temp = pd.read_csv(data_dir_path+'/temperature/0_Global_air_actual_temperatures.csv', low_memory=False)
    df_water_global_temp = pd.read_csv(data_dir_path+'/temperature/0_Global_water_actual_temperatures.csv', low_memory=False)

    # merge all the temperature data for all the cities and global values
    df_temperature = pd.concat(df_list, ignore_index=True, axis=0)
    df_temperature = df_temperature.merge(df_air_global_temp[['Year', 'Month', 'Actual Temperature']],
                                          how='left',
                                          on=['Year', 'Month'],
                                          suffixes=('', '_Air_Global')).merge(
                                        df_water_global_temp[['Year', 'Month', 'Actual Temperature']],
                                        how='left',
                                        on=['Year', 'Month'],
                                        suffixes=('', '_Water_Global'))

    # read co2, population, and volcano data tables
    df_co2 = pd.read_csv(data_dir_path + '/CO2.csv')
    df_population = pd.read_csv(data_dir_path + '/population_modified.csv')
    df_volcano = pd.read_csv(data_dir_path + '/volcano_yearly_VEI_stats.csv')

    # make temperature data compatible for merge by making column names uniform
    df_temperature.rename(columns={'Year': 'year'}, inplace=True)
    df_temperature.rename(columns={'Month': 'month'}, inplace=True)

    # rename column name to make it coherent for better interpretability
    df_co2.rename(columns={'monthly_average': 'monthly_average_co2'}, inplace=True)

    # create volcano data
    df_volcano = df_volcano[df_volcano.year >= 1750]
    df_volcano = df_volcano.groupby(['year']).agg({'year_average_VEI': 'first',
                                                   'year_events_amount': 'first'}).reset_index()

    # adds CO2 data to temperature data
    df_merged_intermediate = pd.merge(df_temperature, df_co2, on=['year', 'month'], how='left')

    # adds population and volcano data
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['year'],
                                                    how='left'), [df_merged_intermediate, df_population, df_volcano])

    # converts the Latitude and Longitude values from string to float
    df_merged["Latitude"] = df_merged['Latitude'].apply(
        lambda row: float(row.split(" ")[0]) if row.split(" ")[1] == "N" else -float(row.split(" ")[0]))
    df_merged["Longitude"] = df_merged['Longitude'].apply(
        lambda row: float(row.split(" ")[0]) if row.split(" ")[1] == "E" else -float(row.split(" ")[0]))

    # print the time taken to build the data
    print(f"Time Taken: {perf_counter() - start_time}")

    return df_merged[(df_merged['year']>=start_year) & (df_merged['year']<=end_year)].reset_index(drop=True)

if __name__ == '__main__':

    # run this file from './src/eda/' location
    print(load_data(start_year=1850, end_year=2020))