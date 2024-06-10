from flask import request, jsonify

from core.Database import Database
from features.train.TrainModel import TrainModel
from flask_app import app
from SQL.SQLQueries import DatabaseOperations as Query


@app.route('/api/train_options', methods=['GET', 'POST'])
def get_data():
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