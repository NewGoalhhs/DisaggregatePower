import datetime

import app
from core.Database import Database

from SQL.SQLQueries import DatabaseOperations as Query
from SQL.SQLQueries import BuildingOperations as BuildingQuery


class PredictModel:
    def __init__(self, model_path):
        path = model_path.split('/')
        self.model_name = path.pop()
        self.model_class = path.pop()

        _module_ = __import__('MachineLearning.' + self.model_class, fromlist=[self.model_class])
        _class_ = getattr(_module_, self.model_class)
        self.model = _class_()
        self.model.load_model(app.__ROOT__ + f"/MachineLearning/models/{model_path}")

    def prepare_predict(self, p):
        # buildings = Database.query(Query.SELECT_ALL.format('Building'))
        # print("The chosen model: ", self.model)
        #
        # p.open_options()
        # for building in buildings:
        #     p.add_option(building['id'], building['name'], building)
        #
        # building = p.choose_option('Choose a building you want to predict: ')
        year, month, day, hour, minute, second = self.get_datetime(p)

        while True:
            input = p.request_input('Enter a datetime (YYYY MM DD H M S): ')
            if input == '':
                break
            split_input = input.split(' ')
            year = int(split_input[0])
            month = int(split_input[1])
            day = int(split_input[2])
            hour = int(split_input[3])
            minute = int(split_input[4])
            second = int(split_input[5])

            if len(split_input) == 6 and year > 0 and 0 < month < 12 and 0 < day < 31 and 0 < hour < 24 and 0 < minute < 60 and 0 < second < 60:
                break
            else:
                p.print_line('Please enter a valid datetime')

        power_usage = 0

        while True:
            input = p.request_input('Enter the power usage amount (W): ')
            try:
                power_usage = int(input)
                break
            except ValueError:
                p.print_line('Please enter a valid power usage amount')

        return year, month, day, hour, minute, second, power_usage

    def predict(self, p):
        year, month, day, hour, minute, second, power_usage = self.prepare_predict(p)
        datetime = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
        data = {
            "datetime": [datetime],
            "power_usage": [power_usage],
            "appliance_in_use": [0]  # dummy value, not used in prediction
        }
        return self.model.predict(data)

    def get_datetime(self, p):
        current = datetime.datetime.now()
        return current.year, current.month, current.day, current.hour, current.minute, current.second

