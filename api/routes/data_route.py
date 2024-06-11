from flask import request, jsonify

from core.Database import Database
from features.predict.PredictModel import PredictModel
from features.train.TrainModel import TrainModel
from flask_app import app
from SQL.SQLQueries import DatabaseOperations as Query


@app.route('/api/train/options', methods=['GET', 'POST'])
def get_train_options():
    models = TrainModel.get_model_options()
    appliances = Database.query(Query.SELECT_ALL.format('Appliance'))

    data = {
        "models": models,
        "appliances": appliances
    }

    if request.method == 'GET':
        return jsonify(data)
    else:
        pass


@app.route('/api/predict/options', methods=['GET', 'POST'])
def get_predict_options():
    data = PredictModel.get_models()

    if request.method == 'GET':
        return jsonify(data)
    else:
        pass