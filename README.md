# SE-Europe-Data_Challenge_Template
Template repository to work with for the NUWE - Schneider Electric European Data Science Challenge in November 2023.

# Tokens:
- b5b8c21b-a637-4e17-a8fe-0d39a16aa849
- fb81432a-3853-4c30-a105-117c86a433ca
- 2334f370-0c85-405e-bb90-c022445bd273
- 1d9cd4bd-f8aa-476c-8cc1-3442dc91506d


#Repo Structure:

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

2. `run_pipeline.sh`




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

```bash
pip install pandas prophet scikit-learn numpy matplotlib


