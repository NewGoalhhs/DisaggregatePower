from core.Database import Database

from SQL.SQLQueries import DatabaseOperations as Query
from SQL.SQLQueries import BuildingOperations as BuildingQuery


class PredictModel:
    def __init__(self, model):
        self.model = model

    def prepare_predict(self, p):
        buildings = Database.query(Query.SELECT_ALL.format('Building'))
        print("The chosen model: ", self.model)

        p.open_options()
        for building in buildings:
            p.add_option(building['building_id'], building['building_name'], self.predict)

        building = p.choose_option('Choose a building you want to predict: ')

        return building

    def predict(self, p):
        prepared_predict = self.prepare_predict(p)

        self.model.predict()
