from flask import request, jsonify

from core.Database import Database
from features.predict.AdvancedPredictModel import AdvancedPredictModel
from features.train.AdvancedTrainModel import AdvancedTrainModel
from features.train.TrainModel import TrainModel
from flask_app import app, socketio
from SQL.SQLQueries import DatabaseOperations as Query
from MachineLearning.advanced.AdvancedPytorchModel import AdvancedPytorchModel
import pandas as pd
import numpy as np


@app.route('/api/train/advanced/options', methods=['GET', 'POST'])
def get_advanced_training_options():
    appliances = Database.query(Query.SELECT_ALL.format('Appliance'))

    data = {
        'appliances': appliances,
        'models': TrainModel.get_model_options()
    }

    if request.method == 'GET':
        return jsonify(data)
    else:
        pass


@app.route('/api/train/advanced/start', methods=['GET', 'POST'])
def start_advanced_training():
    data = request.get_json()

    model = AdvancedPytorchModel(output_size=len(data['appliance_ids']))
    train_model = AdvancedTrainModel(model)

    appliances = train_model.get_appliances(data['appliance_ids'])
    epochs = data['epochs']

    train_model.train(appliances, epochs=epochs)

    if request.method == 'POST':
        return jsonify({"status": "success"})
    else:
        pass


@app.route('/api/predict/advanced/options', methods=['GET', 'POST'])
def get_advanced_predicting_options():
    models = AdvancedPredictModel.get_models()

    data = {
        'models': models
    }

    if request.method == 'GET':
        return jsonify(data)
    else:
        pass


@app.route('/api/predict/advanced/start', methods=['GET', 'POST'])
def start_advanced_predicting():
    if request.method == 'POST':

        data = request.get_json()

        _datetime = data['datetime']
        main_power = [float(x) for x in data['main_power']]
        model = data['model']

        predict_model = AdvancedPredictModel.get_predict_model_from_save_name(model)

        if predict_model is None:
            return jsonify({"status": "error", "message": "Model not found"})

        # Ensure training data is available
        # training_data = AdvancedTrainModel.get_training_data(predict_model.appliances)
        # training_data = pd.DataFrame({})
        # Create a dictionary to hold the new data
        predict_data = {
            'datetime': pd.Series(_datetime),
            'power_usage': pd.Series(main_power)
        }

        # Add appliance_in_use_* columns to the predict_data
        for appliance in predict_model.appliances:
            appliance_column = f'appliance_in_use_{appliance["name"]}'
            predict_data[appliance_column] = pd.concat([
                pd.Series(np.zeros(len(_datetime)))
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
                response[appliance['name']] = {
                    "predictions": prediction[i],
                    "probabilities": probability[i]
                }

        return jsonify({"status": "success", "data": response})
    else:
        return jsonify({"status": "error", "message": "Method not allowed"})
