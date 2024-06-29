# Air Quality Prediction and Classification

This repository contains code for predicting air quality conditions based on various environmental parameters. The primary focus is on predicting PM2.5 levels and categorizing the air quality into different levels based on the predicted values. The project uses a Random Forest Regressor model and includes steps for data preprocessing, model training, hyperparameter tuning, and evaluation.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Description](#model-description)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [License](#license)

## Overview

This project aims to predict PM2.5 levels using various environmental parameters such as temperature, humidity, CO2, NO2, air pressure, and more. It also categorizes the air quality into different levels such as Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, and Hazardous based on the predicted PM2.5 values.

## Dataset

The dataset used in this project should contain the following columns:
- Date
- Time
- Temperature
- Humidity
- PM1.0
- PM2.5
- PM10
- CO2
- CO
- NO2
- Air Pressure
- Altitude
- Latitude
- Longitude

## Installation

To run this project, you need to have Python installed along with several libraries. You can install the required libraries using the following command:

```bash
pip install pandas scikit-learn matplotlib numpy
```
## Usage

1. **Clone the repository**:

```bash
git clone https://github.com/adeeb0005/air-quality-prediction.git
cd air-quality-prediction
```
2. Place your dataset file (air_quality_data.csv) in the repository directory.
3. **Open the Jupyter Notebook:**
```bash
jupyter notebook Air_Quality_Prediction.ipynb
```
4. Run the cells in the notebook to execute the code step by step.

## Model Description

The project uses a Random Forest Regressor to predict PM2.5 levels. The model is trained on a dataset of various environmental parameters. The data is split into training and testing sets, normalized, and then used to train the model. Hyperparameter tuning is performed using GridSearchCV to find the best model parameters.

## Results

The predictions are evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics. The predicted PM2.5 values are categorized into different air quality levels based on predefined thresholds.

## Feature Importance

The feature importance is plotted to show which features have the most significant impact on the prediction of PM2.5 levels.

## Hyperparameter Tuning

GridSearchCV is used for hyperparameter tuning. The parameters tuned include the number of estimators, max depth, min samples split, and min samples leaf for the Random Forest Regressor.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
