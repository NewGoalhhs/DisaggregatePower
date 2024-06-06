from core.Screen import Screen
from features.document.DocumentModel import DocumentModel
from features.train.TrainModel import TrainModel


class DocumentModelScreen(Screen):
    def __init__(self, model):
        super().__init__()
        train_model = TrainModel(model)

        self.document_model = DocumentModel(model)

    def screen(self, p):
        p.print_heading('Document Model')

        # TODO: implementeer PredictModelScreen

        p.print_line('Choose what you want to do:')
        p.open_options()
        p.add_option('1', 'Get interesting datetime', self.document_model.get_interesting_datetime)
        result = p.choose_option()
        print(result)