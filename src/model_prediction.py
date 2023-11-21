# import pandas as pd
# import argparse

# def load_data(file_path):
#     # TODO: Load test data from CSV file
#     return df

# def load_model(model_path):
#     # TODO: Load the trained model
#     return model

# def make_predictions(df, model):
#     # TODO: Use the model to make predictions on the test data
#     return predictions

# def save_predictions(predictions, predictions_file):
#     # TODO: Save predictions to a JSON file
#     pass

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
#     parser.add_argument(
#         '--input_file', 
#         type=str, 
#         default='data/test_data.csv', 
#         help='Path to the test data file to make predictions'
#     )
#     parser.add_argument(
#         '--model_file', 
#         type=str, 
#         default='models/model.pkl',
#         help='Path to the trained model file'
#     )
#     parser.add_argument(
#         '--output_file', 
#         type=str, 
#         default='predictions/predictions.json', 
#         help='Path to save the predictions'
#     )
#     return parser.parse_args()

# def main(input_file, model_file, output_file):
#     df = load_data(input_file)
#     model = load_model(model_file)
#     predictions = make_predictions(df, model)
#     save_predictions(predictions, output_file)

# if __name__ == "__main__":
#     args = parse_arguments()
#     main(args.input_file, args.model_file, args.output_file)




import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
import argparse
import pickle

def load_data(file_path):
    """
    Load test data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def prep_for_prophet(df, index='index', target_col='surplus'):
    """
    Prepare data for use with the Prophet time series forecasting model.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - index (str): Column name to be used as the index.
    - target_col (str): Column name to be used as the target variable.

    Returns:
    - pd.DataFrame: Prepared DataFrame for Prophet.
    """
    df = df.reset_index()
    df[['ds', 'y']] = df[[index, target_col]]
    df.drop([index, target_col], axis=1, inplace=True)
    df.sort_values(by=['ds'], inplace=True)
    return df

def load_model(model_path, country):
    """
    Load a trained Prophet model for a specified country.

    Parameters:
    - model_path (str): Path to the model file.
    - country (str): Country code.

    Returns:
    - Prophet: Loaded Prophet model.
    """
    model_path = f'models/{country}_model.pkl'
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


metrics_list = []  # List to store metrics for each country
result_comparison_list = []  # List to store result comparisons for each country

def make_predictions(df, model, country):
    """
    Make predictions using a trained Prophet model for a specific country.

    Parameters:
    - df (pd.DataFrame): Input DataFrame for the country.
    - model (Prophet): Trained Prophet model.
    - country (str): Country code.

    Returns:
    - Tuple: Lists containing metrics and result comparisons.
    """
    try:
        test_dates = model.make_future_dataframe(periods=len(df), freq='H') ##Hourly prediction
        forecast_base_model = model.predict(test_dates)
        base_model_results = forecast_base_model.iloc[-len(df):]


        
        df_prophet = prep_for_prophet(df, index='floorEndTime', target_col='surplus')
        

        mae = mean_absolute_error(df_prophet['y'], base_model_results['yhat'])
        mse = mean_squared_error(df_prophet['y'], base_model_results['yhat'])
        rmse = np.sqrt(mse)
        
        metrics_list.append({'CountryID': country, 'MAE': mae, 'MSE': mse, 'RMSE': rmse})
        
        result_comparison = pd.DataFrame({'CountryID': country, 'Timestamp': df_prophet['ds'],
                                          'Actual': df_prophet['y'].values, 'Forecast': base_model_results['yhat'].values})
        result_comparison_list.append(result_comparison)


        # Plotting actual vs forecasted values
        plt.figure(figsize=(10, 6))
        plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual', color='blue')
        plt.plot(result_comparison['Timestamp'], result_comparison['Forecast'], label='Forecast', color='red')
        plt.title(f'Actual vs Forecasted Values - {country}')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.legend()

        # Save the plot to a file in the current working directory
        plot_filename = f'./predictions/actual_vs_forecast_{country}.png'
        plt.savefig(plot_filename)

        # Display the plot
        # plt.show()
        # plt.close()

    except Exception as e:
        print(f"Error processing {country}: {str(e)}")

    return metrics_list, result_comparison_list

def surplus(metrics_list, result_comparison_list):
    """
    Calculate surplus metrics and determine the winning countries.

    Parameters:
    - metrics_list (list): List of dictionaries containing metrics for each country.
    - result_comparison_list (list): List of DataFrames containing result comparisons.

    Returns:
    - pd.DataFrame: DataFrame with winning countries and their rankings.
    """
    metrics_df = pd.DataFrame(metrics_list)
    result_comparison_df = pd.concat(result_comparison_list)
    
    print("Metrics DataFrame:")
    print(metrics_df)
    
    winning_indices = result_comparison_df.groupby('Timestamp')['Forecast'].idxmax()
    winning_countries = result_comparison_df.loc[winning_indices].reset_index(drop=True)
    winning_countries = winning_countries.drop_duplicates(subset='Timestamp').reset_index(drop=True)
    
    country_code_mapping = {'SP': 0, 'UK': 1, 'DE': 2, 'DK': 3, 'SE': 4, 'HU': 5, 'IT': 6, 'PO': 7, 'NE': 8}
    winning_countries['Country_Rank'] = winning_countries['CountryID'].map(country_code_mapping)
    
    return winning_countries

def save_predictions(predictions, output_file):
    """
    Save predictions to a JSON file.

    Parameters:
    - predictions (Tuple): Lists containing metrics and result comparisons.
    - output_file (str): Path to save the predictions.

    Returns:
    - None
    """
    winning_countries = surplus(*predictions)
    country_rank_dict = {'target': winning_countries['Country_Rank'].head(442).to_dict()}
    
    json_representation = json.dumps(country_rank_dict, indent=4)
    
    with open(output_file, 'w') as json_file:
        json_file.write(json_representation)

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
    - argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/test_data.csv',
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default=None,
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='predictions/predictions.json',
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    """
    Main function to make predictions.

    Parameters:
    - input_file (str): Path to the test data file.
    - model_file (str): Path to the trained model file.
    - output_file (str): Path to save the predictions.

    Returns:
    - None
    """
    df = load_data(input_file)
    countries = ['DE', 'DK', 'SP', 'UK', 'HU', 'SE', 'IT', 'PO', 'NE']
    predictions = ([], [])  # Initialize empty lists for metrics and result comparisons

    for country in countries:
        try:
            model = load_model(model_file, country)
        except FileNotFoundError:
            print(f"Model not found for {country}. Skipping...")
            continue

        country_df = df[df['CountryID'] == country]
        country_predictions = make_predictions(country_df, model, country)
        predictions = ([*a, *b] for a, b in zip(predictions, country_predictions))

    save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
