from flask import request, jsonify

from core.Database import Database
from features.train.TrainModel import TrainModel
from flask_app import app
from SQL.SQLQueries import DatabaseOperations as Query


@app.route('/api/train/start', methods=['GET', 'POST'])
def start_training():

    data = request.get_json()

    model = TrainModel.get_model(data['model'])
    appliance = Database.query(Query.SELECT_WHERE.format('Appliance', 'id', data['appliance_id']))[0]
    epochs = data['epochs']

    model.train(appliance, epochs=epochs)

    if request.method == 'POST':
        return jsonify({"status": "success"})
    else:
        pass