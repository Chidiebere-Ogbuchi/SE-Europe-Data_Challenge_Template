NUWE-SCHNEIDER Hackathon
Energy Forecasting Challenge
⚙️ Context:
With the increasing digitalization and growing reliance on data servers, the importance of sustainable computing has gained prominence. Schneider Electric, a pioneer in digital transformation and energy management, presents an innovative challenge to contribute to reducing the carbon footprint of the computing industry.

🎯 Objective:
The goal is to develop a model capable of predicting the European country, from a list of nine, that will have the highest surplus of green energy in the next hour. This prediction is crucial for optimizing computing tasks to effectively utilize green energy and subsequently reduce CO2 emissions. The solution should not only align with Schneider Electric's ethos but also introduce an unprecedented approach.

📉 Dataset:
Participants will work with time-series data of hourly granularity extracted from the ENTSO-E Transparency portal using its API: ENTSO-E API Guide. The dataset includes electricity consumption (load), wind energy generation, solar energy generation, and other green energy generation for the countries: Spain, UK, Germany, Denmark, Sweden, Hungary, Italy, Poland, and the Netherlands. These features are aggregated at different intervals (15 min, 30 min, or 1h) depending on the country. The data needs to be homogenized to 1-hour intervals for consistency.

Participants are responsible for using the API to obtain the data. To create 'train.csv' and 'test.csv' datasets, participants should download the data from 01-01-2022 to 01-01-2023, group it as indicated below, and make an 80/20 split. The 80% will be used for training, and the remaining 20% will be used for testing.

📊 Dataset Aggregation:

Electricity consumption (load)
Wind energy generation
Solar energy generation
Other green energy generation

# SE-Europe-Data_Challenge_Template
Template repository to work with for the NUWE - Schneider Electric European Data Science Challenge in November 2023.

# Tokens:
- b5b8c21b-a637-4e17-a8fe-0d39a16aa849
- fb81432a-3853-4c30-a105-117c86a433ca
- 2334f370-0c85-405e-bb90-c022445bd273
- 1d9cd4bd-f8aa-476c-8cc1-3442dc91506d


# Repo Structure:

## Directory Structure Explanation

- **`data/`**: Contains CSV files for training (`train.csv`) and testing (`test.csv`) data.

- **`src/`**: Source code directory containing Python scripts for different pipeline of the project:
  - `data_ingestion.py`: Script for loading and ingesting data.
  - `data_processing.py`: Script for processing and cleaning data.
  - `model_training.py` (or `model_training.ipynb`): Script or notebook for training forecasting models.
  - `model_prediction.py`: Script for making predictions using trained models.
  - `utils.py`: Utility functions used across scripts.

- **`models/`**: Directory to store trained Prophet models (e.g., `{countries}_model.pkl`).

- **`scripts/`**: Contains shell scripts or batch files for running the pipeline. For example, `run_pipeline.sh` might be used to execute the entire workflow.

- **`predictions/`**: Directory to store prediction results. It includes `example_predictions.json` and `predictions.json`.


## Instructions
1. Install dependencies by running:

   ```bash
   pip install -r requirements.txt

2. `run_pipeline.sh` OR
-- python src/data_ingestion.py
-- python src/data_processing.py
-- python src/model_training.py
-- python src/model_prediction.py



# Energy Forecasting Hackathon Prediction Script

## Overview

This script is designed for the Energy Forecasting Hackathon and focuses on making predictions using time series forecasting models, specifically the [Prophet](https://facebook.github.io/prophet/) model. The predictions are made for multiple countries based on trained models, and the results are saved in a JSON file.

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

- [pandas](https://pandas.pydata.org/)
- [prophet](https://facebook.github.io/prophet/)
- [scikit-learn](https://scikit-learn.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

You can install these dependencies using the following command:
pip install pandas prophet scikit-learn numpy matplotlib


# Approach

## Data Ingestion
The first step in our approach is Data Ingestion. We leverage the ENTSO-E Transparency portal API to retrieve time-series data of hourly granularity. The data includes electricity consumption (load), wind energy generation, solar energy generation, and other green energy generation for nine European countries. This API allows us to access real-time and historical data, providing the foundation for our forecasting models.

Period: 01-01-2022 to 01-01-2023,
Countries: Spain, UK, Germany, Denmark, Sweden, Hungary, Italy, Poland, and the Netherlands.
PsrTypes: Green Energy we focused on
| Code | Meaning                           |
|------|-----------------------------------|
| B01  | Biomass                           |
| B09  | Geothermal                        |
| B10  | Hydro Pumped Storage              |
| B11  | Hydro Run-of-river and poundage   |
| B12  | Hydro Water Reservoir             |
| B13  | Marine                            |
| B15  | Other renewable                   |
| B16  | Solar                             |
| B18  | Wind Offshore                     |
| B19  | Wind Onshore                      |


## Data Processing
Following Data Ingestion, the next crucial step is Data Processing. The raw data obtained from the API might be at varying intervals (15 min, 30 min, or 1h) for different countries. To ensure consistency, we homogenize the data to 1-hour intervals. This involves cleaning, aggregating, and structuring the data to create a cohesive dataset for training and testing our models. The processed data is then stored in 'train.csv' and 'test.csv' files. We answer the following?
#### How many data points have we ingested?
#### Do we loose any data during data processing?
#### Which data have we lost?
#### Why did we loose it? However, it is up to you to define the specific measures you are monitoring.

## Model Training
The heart of our approach lies in Model Training. We utilize machine learning techniques, particularly time-series forecasting models **Prophet**, to train on the processed dataset. The goal is to capture the patterns and trends in surplus energy generation for each country. Training is performed separately for each country, allowing the models to learn country-specific nuances.

## Model Prediction
Once the models are trained, we save them in a custom pickle (.pkl) corresponding to each country. Using the trained models, we make predictions for each country on which will have the highest surplus of green energy in the next hour. This prediction is valuable for making informed decisions on optimizing computing tasks to utilize green energy effectively and reduce CO2 emissions. The predictions are then saved in the 'predictions.json' file.

Our approach is designed to align with Schneider Electric's vision for sustainability, aiming not only to predict energy surpluses accurately but also to contribute to reducing the environmental impact of the computing industry. The modular structure of Data Ingestion, Data Processing, Model Training, and Model Prediction allows for flexibility, scalability, and easy integration of new methodologies or models in the future.

#### Other Model recommendations are SARIMA and LSTM



