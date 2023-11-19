import os
import pandas as pd
import argparse

pd.options.mode.chained_assignment = None

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    if 'StartTime' in df.columns and 'EndTime' in df.columns:
        df['StartTime'] = pd.to_datetime(df['StartTime'], format='%Y-%m-%dT%H:%M+00:00Z')
        df['EndTime'] = pd.to_datetime(df['EndTime'], format='%Y-%m-%dT%H:%M+00:00Z')
    return df.dropna()  # Placeholder for cleaning, replace with actual cleaning operations

def preprocess_data(df, data_type):
    # Check if 'load' column exists and create it if it doesn't for 'load' data type
    if data_type == 'load' and 'load' not in df.columns:
        df['load'] = 0
    if 'StartTime' in df.columns and 'EndTime' in df.columns:
        df['Interval'] = (df['EndTime'] - df['StartTime']).dt.total_seconds() / 3600
        df['StartTime'] = df['StartTime'].dt.floor('H')
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

def main(input_folder, output_file):
    # Initialize an empty DataFrame to store aggregated results
    aggregated_data = pd.DataFrame()

    # Loop through files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv') and 'test' not in filename:
            file_path = os.path.join(input_folder, filename)
            print(f'Processing {filename}...')
            # Extract type and country information from the file name
            file_parts = filename.split('.')[0].split('_')
            data_type = file_parts[0]  # Extracting 'gen' or 'load'
            country = file_parts[1]    # Extracting country code
            
            # Read the CSV file
            df = load_data(file_path)
            if 'PsrType' not in df.columns:
                df["PsrType"] = 'LOAD'
            if 'quantity' not in df.columns:
                df["quantity"] = 0
            if 'Load' not in df.columns:
                df["Load"] = 0


            # Add 'type' and 'country' columns based on file name information
            df['DataType'] = data_type
            df['Country'] = country

            # Clean and preprocess the data
            df_clean = clean_data(df)
            df_processed = preprocess_data(df_clean, data_type)
            
            # Append the current DataFrame to the aggregated_data
            aggregated_data = pd.concat([aggregated_data, df_processed])

    # Save the aggregated data to a file
    save_data(aggregated_data, output_file)

    # Group by required columns and sum the quantity while getting the first 'EndTime'
    grouped_data = aggregated_data.groupby(['Country', 'DataType', 'AreaID', 'PsrType', pd.Grouper(key='StartTime', freq='1H')]).agg({
        'quantity': 'sum',
        'Load': 'sum',
        'EndTime': 'first'
    }).reset_index()
    grouped_data["total"] = grouped_data["quantity"].where(grouped_data["quantity"] > 0, grouped_data["Load"])
    # Reorder columns
    grouped_data = grouped_data[['Country', 'DataType', 'AreaID', 'StartTime', 'EndTime', 'PsrType', 'total']]

    # Save the grouped data to an output file
    save_data(grouped_data, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_folder, args.output_file)
