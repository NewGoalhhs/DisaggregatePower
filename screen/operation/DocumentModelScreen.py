from features.document.DocumentModel import DocumentModel
from screen.abstract.AbstractModelScreen import AbstractModelScreen


class DocumentModelScreen(AbstractModelScreen):
    def __init__(self, model):
        super().__init__()
        self.document_model = DocumentModel(model)

    def screen(self, p):
        p.print_heading('Document Model')

        appliance = self.get_appliance(p)
        epochs = self.get_epochs(p)

        model_path = self.document_model.train(appliance, epochs)

        model_path = model_path.replace('E:/Github/DisaggregatePower/MachineLearning/models/', '', 1)

        datetime, power_usage, appliance_in_use = self.document_model.get_interesting_datetime(appliance)

        image_path = self.document_model.predict(model_path, appliance, datetime, power_usage, appliance_in_use, show_plot=False)

        custom_parameters = {
            'epochs': epochs
        }

        self.document_model.generate_document(image_path=image_path, chosen_datetime=datetime[0], appliance=appliance, custom_parameters=custom_parameters)

        p.request_input('Press enter to continue')
