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

    def train(self, appliance, epochs, print_progress: bool = True):
        appliance_id = appliance['id']

        p = PrintHelper()
        lb = p.get_loading_bar(text="Training model", goal=6)

        if print_progress:
            lb.set_status("Querying data")
        power_usage = Database.query(Query.SELECT_ALL.format('PowerUsage'))
        if print_progress:
            lb.update()
        appliance_in_use = Database.query(
            Query.SELECT_WHERE.format('IsUsingAppliance', 'Appliance_id', appliance_id))
        if print_progress:
            lb.update()
        power_usage = pd.DataFrame(power_usage)
        appliance_in_use = pd.DataFrame(appliance_in_use)
        appliance_power_usage_ids = set(appliance_in_use['PowerUsage_id'])
        if print_progress:
            lb.update()

        matches = power_usage['id'].isin(appliance_power_usage_ids)
        power_usage['IsUsingAppliance'] = matches.astype(int)
        if print_progress:
            lb.update()

        # Convert the string power_usage['datetime'] to int for training
        if print_progress:
            lb.update()
        data = {
            "datetime": power_usage.get('datetime'),
            "power_usage": power_usage.get('power_usage'),
            "appliance_in_use": power_usage.get('IsUsingAppliance')
        }
        if print_progress:
            lb.finish()

        self.model.train(data, epochs=epochs)

        # Immediately let the model predict to get a score
        score = self.get_model_score(data)
        if score is not None:
            p.print_line(f"Model score: {score}")
        else:
            score = 1.0

        save_path = self.get_save_path(score, appliance['name'])
        self.model.save_model(save_path)
        if print_progress:
            p.print_line(f"Model saved at {save_path}")

        return save_path

    def get_save_path(self, score, appliance):
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        models_path = app.__ROOT__ + f"/MachineLearning/models/{self.model.__class__.__name__}"

        if not os.path.exists(models_path):
            os.makedirs(models_path)

        score = round(score * 100)

        return f"{models_path}/{datetime_str}_{score}_{appliance}.{self.model.file_extension()}"

    def get_model_score(self, data):
        y = data['appliance_in_use']
        y_pred, _ = self.model.predict(data)
        return self.model.get_score(y, y_pred)

    @staticmethod
    def get_model_options() -> list:
        with os.scandir(app.__ROOT__ + '/MachineLearning') as entries:
            options = []
            index = 0
            for entry in entries:
                if entry.is_dir():
                    continue

                name = entry.name.split('.')[0]

                options.append(name)

                index += 1

            return options

    @staticmethod
    def get_model(model_name: str):
        module = __import__('MachineLearning.' + model_name, fromlist=[model_name])
        model = getattr(module, model_name)
        return TrainModel(model())