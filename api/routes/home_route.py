import datetime
import threading

import pandas as pd
import numpy as np

from SQL.SQLQueries import DatabaseOperations as Query
from core.Database import Database
from features.predict.AdvancedPredictModel import AdvancedPredictModel
from features.predict.PredictModel import PredictModel
from flask_app import socketio

is_running = False

def background_home_content_task():
    threading.Timer(1, background_home_content_task).start()

    response = {}

    power_usage_data = get_power_usage_for_time()

    if power_usage_data is None or len(power_usage_data) == 0:
        return

    power_usage_data = power_usage_data[0]

    for model in PredictModel.get_models():
        predict_model = PredictModel.get_predict_model_from_save_name(model)

        if predict_model is None:
            continue

        power_usage = [power_usage_data]

        power_usage = pd.DataFrame(power_usage)

        # Create a dictionary to hold the new data
        predict_data = {
            'datetime': pd.Series(power_usage['datetime']),
            'power_usage': pd.Series(power_usage['power_usage']),
            'appliance_in_use': pd.Series(np.zeros(len(power_usage['datetime'])))
        }

        # Convert the dictionary to a DataFrame
        predict_data_df = pd.DataFrame(predict_data)

        # Ensure the data is ordered by datetime
        predict_data_df = predict_data_df.sort_values('datetime').reset_index(drop=True)

        # Make predictions
        predictions, probabilities = predict_model.model.predict(predict_data_df)

        # Clip probabilities to be within [0, 1]
        probabilities = [max(min(x, 1), 0) for x in probabilities]

        response[predict_model.appliance['id']] = {
            "name": predict_model.appliance['name'],
            "predictions": predictions[0],
            "probabilities": probabilities[0]
        }

    socketio.emit('home', {
        'power_usage': [power_usage_data['power_usage']],
        'current_datetime': [power_usage_data['datetime']],
        'predictions': response,
    })

def background_home_content_task_advanced():
    threading.Timer(1, background_home_content_task_advanced).start()
    model = "20240618185717_microwave-dishwaser-stove"

    power_usage = get_power_usage_for_time()

    if power_usage is None:
        return

    power_usage = [power_usage[0]]

    power_usage = pd.DataFrame(power_usage)

    predict_model = AdvancedPredictModel.get_predict_model_from_save_name(model)

    if predict_model is None:
        return

    # Ensure training data is available
    # training_data = AdvancedTrainModel.get_training_data(predict_model.appliances)
    # training_data = pd.DataFrame({})
    # Create a dictionary to hold the new data
    predict_data = {
        'datetime': pd.Series(power_usage['datetime']),
        'power_usage': pd.Series(power_usage['power_usage']),
    }

    # Add appliance_in_use_* columns to the predict_data
    for appliance in predict_model.appliances:
        appliance_column = f'appliance_in_use_{appliance["name"]}'
        predict_data[appliance_column] = pd.concat([
            pd.Series(np.zeros(len(power_usage['datetime'])))
        ])

    # Convert the dictionary to a DataFrame
    predict_data_df = pd.DataFrame(predict_data)

    # Ensure the data is ordered by datetime
    predict_data_df = predict_data_df.sort_values('datetime').reset_index(drop=True)

    # Make predictions
    predictions, probabilities = predict_model.model.predict(predict_data_df)

    # Clip probabilities to be within [0, 1]

    probabilities = [[max(min(x, 1), 0) for x in probability] for probability in probabilities]

    response = {}

    for prediction, probability in zip(predictions, probabilities):
        for i, appliance in enumerate(predict_model.appliances):
            response[appliance['id']] = {
                "name": appliance['name'],
                "predictions": prediction[i],
                "probabilities": probability[i]
            }

    socketio.emit('home', {
        'power_usage': power_usage['power_usage'].to_list(),
        'current_datetime': power_usage['datetime'].to_list(),
        'predictions': response,
    })


def get_power_usage_for_time():
    # Pick the current datetime
    current_datetime = datetime.datetime.now()
    current_datetime = current_datetime.replace(year=2011, month=4, day=25)

    # Get a power_usage value from the database related to the current datetime
    power_usage = Database.query(
        Query.SELECT_WHERE.format('PowerUsage', 'datetime', current_datetime.strftime('%Y-%m-%d %H:%M:%S')))

    if len(power_usage) == 0 or power_usage[0]['building_id'] != 1:
        return None

    return power_usage
@socketio.on('home')
def handle_connect(data):
    if data['data'] == 'connected':
        global is_running
        if is_running:
            return
        is_running = True
        background_home_content_task()
