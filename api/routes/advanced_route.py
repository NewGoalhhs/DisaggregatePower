from flask import request, jsonify

from core.Database import Database
from features.train.AdvancedTrainModel import AdvancedTrainModel
from features.train.TrainModel import TrainModel
from flask_app import app
from SQL.SQLQueries import DatabaseOperations as Query
from MachineLearning.advanced.AdvancedPytorchModel import AdvancedPytorchModel


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
