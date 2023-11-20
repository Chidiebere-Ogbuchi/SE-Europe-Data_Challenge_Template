import os
import pandas as pd
import argparse
import numpy as np

pd.options.mode.chained_assignment = None

psrtype_values = ["B01", "B09", "B10", "B11", "B12", "B13", "B15", "B16", "B18", "B19"]

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    if 'StartTime' in df.columns and 'EndTime' in df.columns:
        df['StartTime'] = pd.to_datetime(df['StartTime'], format='%Y-%m-%dT%H:%M+00:00Z')
        df['EndTime'] = pd.to_datetime(df['EndTime'], format='%Y-%m-%dT%H:%M+00:00Z')
    return df

def add_missing_interval_rows(df, minutes=15):

    start_date = pd.to_datetime('2021-12-31 00:45:00')
    end_date = pd.to_datetime('2023-01-01 00:00:00')

    df.drop(columns=['FloorEndTime'], inplace=True)

    # Iterate over each unique 'PsrType'
    for psrtype in df['PsrType'].unique():
        # Filter rows for the specific 'PsrType' and within the date range
        psrtype_df = df[(df['PsrType'] == psrtype)]

        # Check if 'StartTime' is incremental by 15 minutes
        expected_start_time = start_date
        for index, row in psrtype_df.iterrows():
            if row['StartTime'] != expected_start_time:
                # Create new row with incremental time
                new_row = {
                    'StartTime': expected_start_time,
                    'EndTime': expected_start_time + pd.Timedelta(minutes),
                    'AreaID': row['AreaID'],
                    'UnitName': row['UnitName'],
                    'PsrType': row['PsrType'],
                    'quantity': np.nan,  # Set quantity as NaN
                    'CountryID':row['CountryID']
                }
                # Append the new row to the DataFrame
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                # Update expected_start_time for the next iteration
                expected_start_time += pd.Timedelta(minutes)

            # Break and start with the next row
            expected_start_time = row['EndTime']

    # Sort the DataFrame based on 'PsrType' and 'StartTime'
    df.sort_values(by=['PsrType', 'StartTime'], inplace=True, ignore_index=True)

    df['FloorEndTime'] = df['EndTime'].dt.floor('H')

    return df

def impute_values(df):
        
    start_date = pd.to_datetime('2022-01-01 00:00:00')
    end_date = pd.to_datetime('2023-01-01 00:00:00')

    df = df[(df['FloorEndTime'] >= start_date) & (df['FloorEndTime'] <= end_date)]
    try:
        df['quantity'] = df['quantity'].astype(float)
    except ValueError:
        # For typical nans filled with strings replace with nans
        for char in ['-', '--', '?']:
            df['quantity'] = df['quantity'].replace(char, np.nan)

        # Set object type to float
        df['quantity'] = df['quantity'].astype(float)

    # Interpolate missing values in 'quantity'
    df['quantity'].interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
    return df

def drop_additional_columns(df):
    # Drop specified columns
    df.drop(columns=['StartTime', 'EndTime', 'AreaID', 'UnitName'], inplace=True)

    # Filter rows based on the specified Psrtype values
    return df[df['PsrType'].isin(psrtype_values)]

def aggregate_quantity_by_hour(df):
    df = df.drop_duplicates(subset=['PsrType', 'CountryID', 'FloorEndTime'])
    return df.groupby(['PsrType', 'CountryID', 'FloorEndTime'])['quantity'].sum().reset_index()


def preprocess_data(df, minutes, data_type=0):
    # Check if 'load' column exists and create it if it doesn't for 'load' data type
    # if data_type == 'load' and 'load' not in df.columns:
    #     df['load'] = 0

    # df['Interval'] = (df['EndTime'] - df['StartTime']).dt.total_seconds() / 3600

    # Filter data for the specified date range
    start_date = pd.to_datetime('2021-12-31 00:45:00')
    end_date = pd.to_datetime('2023-01-01 00:00:00')
    df = df[(df['StartTime'] >= start_date) & (df['EndTime'] <= end_date)]

    # Temporarily floor it for finding np slots, we drop it instantly
    df['FloorEndTime'] = df['EndTime'].dt.floor('H')

    df = add_missing_interval_rows(df, minutes)
    df = impute_values(df)
    df = drop_additional_columns(df)

    return df

def save_data(df, output_file):
    df.to_csv(output_file, index=False)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_folder',
        type=str,
        default='./data',
        help='Path to the folder containing input CSV files'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='./processed_data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_folder='./data/gen_data/', output_dir='./processed_data/'):
    # Initialize an empty DataFrame to store aggregated results
    aggregated_data = pd.DataFrame()

    # Loop through files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            # Extract type and country information from the file name
            file_parts = filename.split('.')[0].split('_')
            data_type = file_parts[0]  # Extracting 'gen' or 'load'
            country = file_parts[1]    # Extracting country code
            psrtype = file_parts[2]    # Extracting country code
            if not psrtype in psrtype_values:
                print(f'{psrtype} Skipping {filename}...')
                continue
            else:
                print(f'Processing {filename}...')
            
            # Read the CSV file
            df = load_data(file_path)

            # if 'PsrType' not in df.columns:
            #     df["PsrType"] = 'LOAD'
            # if 'quantity' not in df.columns:
            #     df["quantity"] = 0
            # if 'Load' not in df.columns:
            #     df["Load"] = 0


            # Add 'type' and 'country' columns based on file name information
            # df['DataType'] = data_type
            df['CountryID'] = country
            
            # Clean and preprocess the data
            df = clean_data(df)
            # NE in filenames
            if country in ['DE', 'SP', 'NE', 'HU']: # 15min
                df = preprocess_data(df, minutes=15)
            elif country in ['UK']: # 30min
                df = preprocess_data(df, minutes=30)
            elif country in ['IT', 'DK', 'PO', 'SE']: # 1h
                df = preprocess_data(df, minutes=60)
            else:
                raise Exception(f'{country} Country not found in the list')
            
            df = aggregate_quantity_by_hour(df)
            # Append the current DataFrame to the aggregated_data
            aggregated_data = pd.concat([aggregated_data, df])

    # print(aggregated_data)
    if(aggregated_data.shape[0] > 0):
        for country_id, group_data in aggregated_data.groupby('CountryID'):
            # Formulate the output file path based on the country_id
            output_file = os.path.join(output_dir, f'{country_id}.csv')
            
            # Save the group_data to the file
            save_data(group_data, output_file)
        print('Process Completed, file saved')
    else:
        print('Empty DataFrame')

    # Group by required columns and sum the quantity while getting the first 'EndTime'
    # grouped_data = aggregated_data.groupby(['Country', 'DataType', 'AreaID', 'PsrType', pd.Grouper(key='StartTime', freq='1H')]).agg({
    #     'quantity': 'sum',
    #     'Load': 'sum',
    #     'EndTime': 'first'
    # }).reset_index()
    # grouped_data["total"] = grouped_data["quantity"].where(grouped_data["quantity"] > 0, grouped_data["Load"])
    # # Reorder columns
    # grouped_data = grouped_data[['Country', 'DataType', 'AreaID', 'StartTime', 'EndTime', 'PsrType', 'total']]

    # # Save the grouped data to an output file
    # save_data(grouped_data, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    # main(args.input_folder, args.output_file)
    main()
    # main('./data/ggen/', './data/ggen/process/')
