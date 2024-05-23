import os
from datetime import datetime

import pandas as pd

from SQL.SQLQueries import DatabaseOperations as Query
from core.Database import Database

import app
from helper.PrintHelper import PrintHelper


class TrainModel:
    def __init__(self, model):
        self.model = model

    def train(self, p: PrintHelper):
        appliance = self.prepare_train(p)
        appliance_id = appliance['id']
        lb = p.get_loading_bar(text="Training model", goal=6)
        lb.set_status("Querying data")
        power_usage = Database.query(Query.SELECT_ALL.format('PowerUsage'))
        lb.update()
        appliance_in_use = Database.query(
            Query.SELECT_WHERE.format('IsUsingAppliance', 'Appliance_id', appliance_id))
        lb.update()
        power_usage = pd.DataFrame(power_usage)
        appliance_in_use = pd.DataFrame(appliance_in_use)
        appliance_power_usage_ids = set(appliance_in_use['PowerUsage_id'])
        lb.update()

        matches = power_usage['id'].isin(appliance_power_usage_ids)
        power_usage['IsUsingAppliance'] = matches.astype(int)
        lb.update()

        # Convert the string power_usage['datetime'] to int for training
        power_usage['datetime'] = power_usage['datetime'].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())
        lb.update()
        data = {
            "timestamp": power_usage.get('datetime').values,
            "power_usage": power_usage.get('power_usage').values,
            "appliance_in_use": power_usage.get('IsUsingAppliance').values
        }
        lb.finish()

        self.model.train(data)

        # Immediately let the model predict to get a score
        score = self.get_model_score(data)

        self.model.save_model(self.get_save_path(score, appliance['name']))

    def prepare_train(self, p):
        appliances = Database.query(Query.SELECT_ALL.format('Appliance'))
        p.print_line("All appliances:")
        p.open_options()
        for index, appliance in enumerate(appliances):
            p.add_option(index+1, appliance['name'], appliance)
        appliance = p.choose_option('Choose an appliance to train: ')
        return appliance

    def get_save_path(self, score, appliance):
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        models_path = app.__ROOT__ + f"/MachineLearning/models/{self.model.__class__.__name__}"

        if not os.path.exists(models_path):
            os.makedirs(models_path)

        score = round(score * 100)

        return f"{models_path}/{datetime_str}_{score}_{appliance}.{self.model.file_extension()}"

    def get_model_score(self, data):
        X, y = self.model.preprocess_data(data)
        y_pred = self.model.predict(X)
        return self.model.get_score(y, y_pred)