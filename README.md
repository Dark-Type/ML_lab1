# Spaceship Titanic Classifier

**Name:** Skazhutin Nikita 
**Group:** 972301

This repository contains an implementation of a machine learning model to predict passenger transport in the Spaceship Titanic challenge.

## How To Use

### Setup

1. Clone this repository:
git clone https://github.com/Dark-Type/ML_lab1.git cd ML_lab1
2. Install using Poetry:
poetry install
   Or install from wheel file:
pip install spaceship_classifier-0.1.0-py3-none-any.whl

### Training a Model

To train the model:
poetry run python -m spaceship_classifier.model train --dataset=./data/train.csv 


This will:
- Process the dataset
- Perform feature engineering
- Optimize hyperparameters using Optuna
- Train the final XGBoost model
- Save the model to `./model/model.pkl`
- Log metrics to `./data/log_file.log`

### Making Predictions

To generate predictions:
poetry run python -m spaceship_classifier.model predict --dataset=./data/test.csv


This will:
- Load the trained model
- Process the test dataset with the same feature engineering
- Generate predictions
- Save results to `./data/results.csv`

