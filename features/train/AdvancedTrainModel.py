import os
from datetime import datetime

import pandas as pd

from SQL.SQLQueries import DatabaseOperations as Query
from core.Database import Database

import app
from helper.PrintHelper import PrintHelper


class AdvancedTrainModel:
    def __init__(self, model):
        self.model = model

    def train(self, appliances, epochs, print_progress: bool = True):

        data = self.get_training_data(appliances)

        self.model.train(data, epochs=epochs)

        # Immediately let the model predict to get a score
        score = self.get_model_score(data)

        save_path = self.get_save_path(score, appliances)
        self.model.save_model(save_path)

        return save_path

    def get_save_path(self, score, appliance):
        models_path = app.__ROOT__ + f"/MachineLearning/advanced/models/{self.model.__class__.__name__}"

        if not os.path.exists(models_path):
            os.makedirs(models_path)

        appliance_str = '_'.join([app['name'] for app in appliance])

        datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")

        return f"{models_path}/{datetime_str}_{appliance_str}.{self.model.file_extension()}"

    def get_model_score(self, data):
        y = data['appliance_in_use']
        y_pred, _ = self.model.predict(data)
        return self.model.get_score(y, y_pred)

    @staticmethod
    def get_model_options() -> list:
        with os.scandir(app.__ROOT__ + '/MachineLearning/advanced') as entries:
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
        module = __import__('MachineLearning.advanced.' + model_name, fromlist=[model_name])
        model = getattr(module, model_name)
        return AdvancedTrainModel(model())

    @classmethod
    def get_training_data(cls, appliances):
        # Initialize an empty DataFrame with the required columns
        data = pd.DataFrame(
            columns=["datetime", "power_usage"] + ["appliance_in_use_" + appliance['name'] for appliance in appliances])

        for appliance in appliances:
            appliance_data = cls.get_appliance_data(appliance)
            appliance_df = pd.DataFrame(appliance_data)
            appliance_df.rename(columns={"appliance_in_use": "appliance_in_use_" + appliance['name']}, inplace=True)

            # Merge the current appliance data with the main data on 'datetime'
            if data.empty:
                data = appliance_df
            else:
                data = pd.merge(data, appliance_df, on="datetime", how="outer", suffixes=('', '_duplicate'))

                # Combine power_usage columns if duplicates exist
                if "power_usage_duplicate" in data.columns:
                    data["power_usage"] = data[["power_usage", "power_usage_duplicate"]].sum(axis=1)
                    data.drop(columns=["power_usage_duplicate"], inplace=True)

        # Fill NaN values with 0 or an appropriate default
        data.fillna(0, inplace=True)

        # Convert DataFrame back to dictionary
        result = data.to_dict('list')

        return result

    @classmethod
    def get_appliance_data(cls, appliance):
        appliance_id = appliance['id']

        power_usage = Database.query(Query.SELECT_ALL.format('PowerUsage'))

        appliance_in_use = Database.query(
            Query.SELECT_WHERE.format('IsUsingAppliance', 'Appliance_id', appliance_id))

        power_usage = pd.DataFrame(power_usage)
        appliance_in_use = pd.DataFrame(appliance_in_use)
        appliance_power_usage_ids = set(appliance_in_use['PowerUsage_id'])

        matches = power_usage['id'].isin(appliance_power_usage_ids)
        power_usage['IsUsingAppliance'] = matches.astype(int)

        return {
            "datetime": power_usage.get('datetime'),
            "power_usage": power_usage.get('power_usage'),
            "appliance_in_use": power_usage.get('IsUsingAppliance')
        }

    @staticmethod
    def get_appliances(appliance_ids):
        appliances = []
        for appliance_id in appliance_ids:
            appliance = Database.query(Query.SELECT_WHERE.format('Appliance', 'id', appliance_id))
            if appliance:
                appliances.append(appliance[0])
        return appliances
