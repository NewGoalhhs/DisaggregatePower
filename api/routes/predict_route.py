import datetime

import numpy as np
import pandas as pd
from flask import request, jsonify

from core.Database import Database
from features.predict.PredictModel import PredictModel
from features.train.TrainModel import TrainModel
from flask_app import app
from SQL.SQLQueries import DatabaseOperations as Query


@app.route('/api/predict/start', methods=['GET', 'POST'])
def start_predicting():
    if request.method == 'POST':

        data = request.get_json()

        _datetime = data['datetime']
        main_power = [float(x) for x in data['main_power']]

        response = {}

        for model in PredictModel.get_models():
            predict_model = PredictModel.get_predict_model_from_save_name(model)

            if predict_model is None:
                data[model] = {"error": "Model not found"}
            predict_data = {
                'datetime': pd.Series(_datetime),
                'power_usage': pd.Series(main_power),
                'appliance_in_use': pd.Series(np.zeros(len(_datetime)))
            }

            predictions, probabilities = predict_model.model.predict(predict_data)

            probabilities = [max(min(x, 1), 0) for x in probabilities]

            response[model] = {
                "predictions": predictions,
                "probabilities": probabilities
            }

        return jsonify({"status": "success", "data": response})
    else:
        return jsonify({"status": "error", "message": "Method not allowed"})