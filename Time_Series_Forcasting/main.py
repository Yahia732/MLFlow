# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from flask import Flask, request, jsonify
import mlflow
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from mlflow.tracking import MlflowClient
import os
import pickle
from statsmodels.tsa.stattools import acf, pacf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
import warnings

warnings.simplefilter("ignore")

app = Flask(__name__)


def train_model():
    directory_path = 'train_splits'
    for filename in os.listdir(directory_path):
        mlflow.end_run()
        mlflow.start_run()
        full_filepath = os.path.join(directory_path, filename)
        dataset_number = filename.split('.')[0].split('_')[1]

        data = pd.read_csv(full_filepath)
        data = data[['timestamp', 'value']]
        data.dropna(inplace=True)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        total_seconds = (data['timestamp'].iloc[1] - data['timestamp'].iloc[0]).total_seconds()
        #file_path = "Periods/" + str(dataset_number)


        mlflow.log_param("Periods", str(total_seconds))


        stl_result = STL(data['value'],
                         period=max(pd.to_timedelta(total_seconds, unit='s') / pd.Timedelta(days=1), 2)).fit()
        data['Trend'] = stl_result.trend
        data['Seasonality'] = stl_result.seasonal
        data['Noise'] = stl_result.resid
        date_feature = pd.to_datetime(data["timestamp"]).dt
        data['month'] = date_feature.month
        data['day'] = date_feature.day
        data['hour'] = date_feature.hour
        data['minute'] = date_feature.minute
        data['day_of_week'] = date_feature.dayofweek
        residual_acf = acf(data['value'], nlags=40)

        # Set a threshold for autocorrelation
        threshold = 0.2  # You can adjust this threshold based on your criteria

        # Find the first lag where autocorrelation is below the threshold
        best_lags = np.argmax(np.abs(residual_acf) < threshold)
        if best_lags == 0:
            best_lags = 1
        #file_path = "Lags/" + str(dataset_number)


        mlflow.log_param("Lags", str(best_lags))



        for i in range(1, best_lags + 1):
            data[f'value_lag_{i}'] = data['value'].shift(i)
            data[f'trend_lag_{i}'] = data['Trend'].shift(i)
            data[f'seasonality_lag_{i}'] = data['Seasonality'].shift(i)
            data[f'noise_lag_{i}'] = data['Noise'].shift(i)

        # Drop rows with NaN resulting from the shift
        data = data.dropna()
        tscv = TimeSeriesSplit(n_splits=2)
        for train_index, val_index in tscv.split(data):
            train_data, val_data = data.iloc[train_index], data.iloc[val_index]
        X_train = train_data.drop(columns=['value', 'timestamp', 'Trend', 'Seasonality', 'Noise'])
        y_train = train_data['value']

        # Extract features and target from the validation set
        X_val = val_data.drop(columns=['value', 'timestamp', 'Trend', 'Seasonality', 'Noise'])

        y_val = val_data['value']

        # Build a polynomial regression model
        degree = 2  # Choose the degree of the polynomial
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(X_train)
        X_val_poly = poly_features.transform(X_val)

        # X_train_poly=np.concatenate([X_train_poly,train_data['timestamp'].values.reshape(train_data.shape[0],1)])
        # X_val_poly=np.concatenate([X_val_poly, val_data['timestamp'].values.reshape(val_data.shape[0],1)])

        model = LinearRegression()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'Models/' +  str(dataset_number))
        mlflow.set_tag("mlflow.runName",str(dataset_number))
        # with open('Models/' + 'model_' + str(dataset_number) + '.pkl', 'wb') as model_file:
        #     pickle.dump(model, model_file)

        # Make predictions on the validation set
        y_val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)


def preprocess_data(data, dataset_id):
    # Use a breakpoint in the code line below to debug your script.
    data = data[['time', 'value']]
    data.dropna(inplace=True)
    data['time'] = pd.to_datetime(data['time'])
    # Specify the experiment name (you can set this to the desired experiment)


    client = MlflowClient()
    id=str(dataset_id)


    # Set the SQLite URI as the backend store
    try:
        filter_string = f"tags.mlflow.runName='{id}'"
        runs = list(client.search_runs(
            experiment_ids=["0"],
            filter_string=filter_string,
            order_by=["start_time desc"],
            max_results=1
        ))

        if not runs:
            raise ValueError(f'No runs found for model {dataset_id}. Filter: {filter_string}')

        run_info = runs[0]
        run_id = run_info.info.run_id


        # Retrieve additional parameters from the run details
        run_details = client.get_run(run_id)
        best_lags = run_details.data.params.get("Lags")
        total_seconds = run_details.data.params.get("Periods")

        seconds_to_add = total_seconds




    except Exception as e:
        print(f"Error retrieving run information: {str(e)}")







    try:

        period = pd.to_timedelta(float(total_seconds), unit='s')


        stl_result = STL(data['value'], period=max(period / pd.Timedelta(days=1), 2)).fit()
    except ValueError as ve:
        return f"Error converting total_seconds to float: {ve}"
    data['Trend'] = stl_result.trend
    data['Seasonality'] = stl_result.seasonal
    data['Noise'] = stl_result.resid




    last_timestamp = data['time'].iloc[-1]
    new_timestamp = last_timestamp + timedelta(seconds=float(seconds_to_add))

    new_row = {'time': new_timestamp, 'value': None, 'Trend': None, 'Seasonality': None, 'Noise': None}
    data.loc[len(data)] = new_row

    date_feature = pd.to_datetime(data["time"]).dt
    data['month'] = date_feature.month
    data['day'] = date_feature.day
    data['hour'] = date_feature.hour
    data['minute'] = date_feature.minute
    data['day_of_week'] = date_feature.dayofweek


    for i in range(1, int(best_lags) + 1):
        data[f'value_lag_{i}'] = data['value'].shift(i)
        data[f'trend_lag_{i}'] = data['Trend'].shift(i)
        data[f'seasonality_lag_{i}'] = data['Seasonality'].shift(i)
        data[f'noise_lag_{i}'] = data['Noise'].shift(i)
    data = data.drop(columns=['value', 'time', 'Trend', 'Seasonality', 'Noise'])
    # Drop rows with NaN resulting from the shift
    data.dropna(inplace=True)

    return data


def load_model(data, dataset_id):
    # Set the SQLite URI as the backend store
    client = MlflowClient()

    id=str(dataset_id)
    print(id)

    # Set the SQLite URI as the backend store
    try:
        filter_string = f"tags.mlflow.runName='{id}'"
        runs = list(client.search_runs(
            experiment_ids=["0"],
            filter_string=filter_string,
            order_by=["start_time desc"],
            max_results=1
        ))

        if not runs:
            raise ValueError(f'No runs found for model {str(dataset_id)}. Filter: {filter_string}')

        run_info = runs[0]
        run_id = run_info.info.run_id
        print(run_id)
    except Exception as e:
        print(f"Error retrieving run information: {str(e)}")
        # Retrieve additional parameters from the run details

    model_path = f"./mlruns/0/{str(run_id)}/artifacts/models/{str(dataset_id)}"
    model = mlflow.sklearn.load_model(model_path)

    # Load the model back using MLflow from the specified path



    # with open('Models/' + 'model_' + str(dataset_id) + '.pkl', 'rb') as model_file:
    #     model = pickle.load(model_file)

    # Make predictions on the validation set
    # Assuming 'data' is the input for the model prediction
    print(data)
    y_val_pred = model.predict(data)
    return y_val_pred[-1]


# Press the green button in the gutter to run the script.
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request
        request_data = request.get_json()

        # Get dataset ID and values from the request
        dataset_id = request_data['dataset_id']
        values = request_data['values']
        data = pd.DataFrame(values)

        # Load the corresponding model based on the dataset ID

        data = preprocess_data(data, dataset_id)
        prediction = load_model(data, dataset_id)

        # Return the prediction in the response body
        response_body = {'output': prediction}
        return jsonify(response_body)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    # mlflow.start_run()
    # train_model()
    app.run(debug=True)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
