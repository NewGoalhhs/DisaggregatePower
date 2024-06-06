from core.Database import Database
from core.Screen import Screen
from features.document.DocumentModel import DocumentModel
from features.train.TrainModel import TrainModel
from SQL.SQLQueries import DatabaseOperations as Query
from screen.abstract.AbstractModelScreen import AbstractModelScreen


class DocumentModelScreen(AbstractModelScreen):
    def __init__(self, model):
        super().__init__()
        self.train_model = TrainModel(model)

    def screen(self, p):
        p.print_heading('Train Model')

        appliance = self.get_appliance(p)
        epochs = self.get_epochs(p)

        self.train_model.train(appliance, epochs)

        p.request_input('Press enter to continue')