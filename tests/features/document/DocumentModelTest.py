import os
from abc import ABC
from app import __ROOT__

from MachineLearning.PytorchModel import PytorchModel
from core.Test import Test
from features.document.DocumentModel import DocumentModel


class DocumentModelTest(Test, ABC):
    def __init__(self):
        super().__init__()
        self.appliance = {'id': '6', 'name': 'microwave'}
        # scan the models directory for the model file
        self.model_path = ''

        with os.scandir(__ROOT__ + '/MachineLearning/models/PytorchModel') as entries:
            for entry in entries:
                if self.appliance['name'] in entry.name:
                    self.model_path = 'PytorchModel/' + entry.name

        self.model = PytorchModel()
        self.document_model = DocumentModel(self.model)

    def test_get_interesting_datetime(self):
        datetime, power_usage, appliance_in_use = self.document_model.get_interesting_datetime(self.appliance)
        self.assert_true(len(datetime) > 0, 'Datetime is empty')
        self.assert_true(len(power_usage) > 0, 'Power usage is empty')
        self.assert_true(len(appliance_in_use) > 0, 'Appliance in use is empty')

    def test_generate_document(self):
        datetime, power_usage, appliance_in_use = self.document_model.get_interesting_datetime(self.appliance)
        image_path = self.document_model.predict(self.model_path, self.appliance, datetime, power_usage, appliance_in_use, show_plot=False, print_progress=False)
        self.document_model.generate_document(image_path=image_path, chosen_datetime=datetime[0], appliance=self.appliance['name'])