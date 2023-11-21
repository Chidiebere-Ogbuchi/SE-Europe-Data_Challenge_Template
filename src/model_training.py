# import pandas as pd
# import argparse

# def load_data(file_path):
#     # TODO: Load processed data from CSV file
#     return df

# def split_data(df):
#     # TODO: Split data into training and validation sets (the test set is already provided in data/test_data.csv)
#     return X_train, X_val, y_train, y_val

# def train_model(X_train, y_train):
#     # TODO: Initialize your model and train it
#     return model

# def save_model(model, model_path):
#     # TODO: Save your trained model
#     pass

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
#     parser.add_argument(
#         '--input_file', 
#         type=str, 
#         default='data/processed_data.csv', 
#         help='Path to the processed data file to train the model'
#     )
#     parser.add_argument(
#         '--model_file', 
#         type=str, 
#         default='models/model.pkl', 
#         help='Path to save the trained model'
#     )
#     return parser.parse_args()

# def main(input_file, model_file):
#     df = load_data(input_file)
#     X_train, X_val, y_train, y_val = split_data(df)
#     model = train_model(X_train, y_train)
#     save_model(model, model_file)

# if __name__ == "__main__":
#     args = parse_arguments()
#     main(args.input_file, args.model_file)



import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import argparse
import pickle


def load_data(file_path):
    """
    Load data from a CSV file, preprocess it, and filter relevant information.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Processed and filtered DataFrame.
    """
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    # df = df.drop(columns=['AreaID', 'EndTime'])

    # Filter countries and PsrTypes
    countries = ['DE', 'DK', 'SP', 'UK', 'HU', 'SE', 'IT', 'PO', 'NE']
    PsrTypes = ['B01', 'B09', 'B10', 'B11', 'B12', 'B13', 'B15', 'B16', 'B18', 'B19', 'AA']
    df = df[df['CountryID'].isin(countries) & df['PsrType'].isin(PsrTypes)]

    # Convert 'floorEndTime' to datetime
    df['floorEndTime'] = pd.to_datetime(df['floorEndTime'])

    # Change PsrType to 'BB' for rows where it is not 'AA'
    df.loc[df['PsrType'] != 'AA', 'PsrType'] = 'BB'

    # Group by specified columns and aggregate quantity
    df = df.groupby(['floorEndTime', 'PsrType', 'CountryID']).agg({'quantity': 'sum'}).reset_index()

    # Create separate DataFrames for 'AA' and 'BB'
    aa_df = df[df['PsrType'] == 'AA']
    bb_df = df[df['PsrType'] == 'BB']

    # Merge the DataFrames on 'floorEndTime', 'CountryID'
    merged_df = pd.merge(aa_df, bb_df, on=['floorEndTime', 'CountryID'], suffixes=('_AA', '_BB'), how='outer')

    # Calculate surplus as the difference between 'BB' and 'AA' quantities
    merged_df['surplus'] = merged_df['quantity_BB'] - merged_df['quantity_AA']
    df = merged_df
    return df



def split_data(df):
    """
    Split data into training and testing sets based on a specific date.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Training DataFrame.
    """
    start_split_date = '2022-01-01'
    end_split_date = '2022-04-09'
    data = df.loc[(df['floorEndTime'] >= start_split_date) & (df['floorEndTime'] < end_split_date)]
    total_days = (data['floorEndTime'].max() - data['floorEndTime'].min()).days
    split_date = data['floorEndTime'].min() + pd.to_timedelta(0.8 * total_days, unit='D')
    train = data[data['floorEndTime'] <= split_date]
    test = data[data['floorEndTime'] > split_date]
    test_data_path = './data/test_data.csv'
    test.to_csv(test_data_path, index=False)
    print(f"Training Data Size: {len(train)}, Test Data Size: {len(test)}")
    return train

def prep_for_prophet(data, index='index', target_col='surplus'):
    """
    Prepare data for use with the Prophet time series forecasting model.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.
    - index (str): Column name to be used as the index.
    - target_col (str): Column name to be used as the target variable.

    Returns:
    - pd.DataFrame: Prepared DataFrame for Prophet.
    """
    data = data.reset_index()
    data[['ds', 'y']] = data[[index, target_col]]
    data.drop([index, target_col], axis=1, inplace=True)
    data.sort_values(by=['ds'], inplace=True)
    return data

def train_model(train):
    """
    Train Prophet models for each country and store them in a dictionary.

    Parameters:
    - train (pd.DataFrame): Training DataFrame.

    Returns:
    - dict: Dictionary containing trained Prophet models for each country.
    """
    trained_models = {}
    countries = ['DE', 'DK', 'SP', 'UK', 'HU', 'SE', 'IT', 'PO', 'NE']

    for country in countries:
        try:
            country_data = train[train['CountryID'] == country]
            prophet_data = prep_for_prophet(country_data, index='floorEndTime', target_col='surplus')
            model = Prophet()
            model.fit(prophet_data)
            trained_models[country] = model
            print(f"Successfully trained model for {country}")

        except Exception as e:
            print(f"Error processing {country}: {str(e)}")
            continue

    return trained_models

def save_models(models):
    """
    Save trained Prophet models to disk.

    Parameters:
    - models (dict): Dictionary containing trained Prophet models.

    Returns:
    - None
    """
    for country, model in models.items():
        model_path = f'models/{country}_model.pkl'
        save_model(model, model_path)
        print(f"Model for {country} saved to {model_path}")

def save_model(model, model_path):
    """
    Save a single trained Prophet model to disk.

    Parameters:
    - model: Trained Prophet model.
    - model_path (str): Path to save the model.

    Returns:
    - None
    """
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
    - argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/final_preprocessed_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file):
    """
    Main function to execute the training process.

    Parameters:
    - input_file (str): Path to the processed data file.
    - model_file (str): Path to save the trained model.

    Returns:
    - None
    """
    df = load_data(input_file)
    train = split_data(df)
    models = train_model(train)
    save_models(models)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)
