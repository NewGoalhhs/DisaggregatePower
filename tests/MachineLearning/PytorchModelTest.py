import pandas as pd

import app
from MachineLearning.PytorchModel import PytorchModel
from core.Test import Test
from features.predict.PredictModel import PredictModel


class PytorchModelTest(Test):

    def __init__(self):
        super().__init__()
        self.appliance = {'id': '6', 'name': 'test'}
        self.model_path = app.__ROOT__ + '/tests/fake/model/test.pt'
        self.image_path = app.__ROOT__ + '/tests/out/test.png'

    def test_train(self):
        df = pd.read_csv(app.__ROOT__ + '/QuickRunData/synthetic_power_usage_data.csv')
        data = {
            'datetime': df['datetime'],
            'power_usage': df['power_usage'],
            'appliance_in_use': df['appliance_in_use']
        }
        model = PytorchModel()
        model.train(data)
        probabilities, predictions = model.predict(data)

        PredictModel.visualize(predictions, data['appliance_in_use'], probabilities, self.image_path)