import datetime
import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import app
from core.Database import Database

from SQL.SQLQueries import DatabaseOperations as Query
from SQL.SQLQueries import PowerUsageOperations as PowerUsageQuery
from SQL.SQLQueries import IsUsingApplianceOperations as IsUsingApplianceQuery
from helper.PrintHelper import PrintHelper


class PredictModel:
    def __init__(self, model_path, appliance):
        path = model_path.replace('\\', '/').split('/')
        self.appliance = appliance
        self.model_name = path.pop()
        self.model_class = path.pop()

        _module_ = __import__('MachineLearning.' + self.model_class, fromlist=[self.model_class])
        _class_ = getattr(_module_, self.model_class)
        self.model = _class_()
        try:
            self.model.load_model(app.__ROOT__ + f"/MachineLearning/models/{model_path}")
        except RuntimeError:
            print("Model not working. " + app.__ROOT__ + f"/MachineLearning/models/{model_path}")

    def predict(self, datetime, power_usage, appliance_in_use):

        data = {
            "datetime": datetime,
            "power_usage": power_usage,
            "appliance_in_use": appliance_in_use
        }

        predictions, propabilities = self.model.predict(data)

        return predictions, propabilities

    def get_datetime(self, p):
        current = datetime.datetime.now()
        return current.year, current.month, current.day, current.hour, current.minute, current.second

    def use_existing_data(self, datetime):
        # Get the next minute from the random datetime
        next_minute = datetime.datetime.strptime(datetime, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(minutes=1)
        # Get the power usage from the random datetime to the next minute
        power_usage = Database.query(PowerUsageQuery.SELECT_POWER_USAGE_BETWEEN.format(datetime, next_minute))
        if len(power_usage) == 0:
            exit()
        else:
            date_time = [data['datetime'] for data in power_usage]
            main_power = [data['power_usage'] for data in power_usage]

            appliance_in_use = []
            for power_usage_i in power_usage:
                appliance_in_use.append(bool(Database.query(
                    IsUsingApplianceQuery.SELECT_WHERE_POWERUSAGE_FOR_APPLIANCE.format(power_usage_i['id'],
                                                                                       self.appliance['id']))))

            return date_time, main_power, appliance_in_use

    @classmethod
    def visualize(cls, predictions, real_data, propabilities, image_path, show_plot=True):
        # Preprocess data

        # Convert tensors to numpy arrays for plotting
        real_data = np.array(real_data)
        predictions = np.array(predictions)
        propabilities = np.array(propabilities)

        # Plot the actual vs predicted values
        plt.figure(figsize=(14, 7))
        plt.plot(real_data, label='Actual Is Using Appliance', alpha=0.75)
        plt.plot(predictions, label='Predicted Is Using Appliance', alpha=0.75)
        plt.plot(propabilities, label='Propability Is Using Appliance', alpha=0.75)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Appliance Usage (0/1)')
        plt.title('Actual vs Predicted Appliance Usage')
        plt.legend()

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path)
        if show_plot:
            plt.show()
        return image_path

    @staticmethod
    def get_models() -> list:
        # Return a list of indexes and generate class options from features/generate
        with os.scandir('MachineLearning/models') as entries:
            options = []
            for entry in entries:
                if entry.is_dir():
                    with os.scandir('MachineLearning/models/' + entry.name) as sub_entries:
                        for index, sub_entry in enumerate(sub_entries):
                            model = sub_entry.name.split('.')[0]

                            options.append(model)
            return options

    def get_image_path(self):
        datetime_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return app.__ROOT__ + f"/src/{self.model_class}/{datetime_str}_{self.appliance['name']}.png"


    @staticmethod
    def get_predict_model_from_save_name(save_name):
        with os.scandir('MachineLearning/models') as entries:
            for entry in entries:
                if entry.is_dir():
                    with os.scandir('MachineLearning/models/' + entry.name) as sub_entries:
                        for sub_entry in sub_entries:
                            sub_entry_name = sub_entry.name.split('.')[0]
                            if sub_entry_name == save_name:
                                appliance = PredictModel.get_appliance(sub_entry.name)
                                return PredictModel(entry.name + '/' + sub_entry.name, appliance)
        return None

    @staticmethod
    def get_appliance(file_name, additional=''):
        if not file_name:
            return ''

        split_entry = file_name.split('.')[0]
        appliance_name = split_entry.split('_')[-1] + additional
        appliance = Database.query(Query.SELECT_WHERE.format('Appliance', 'name', appliance_name))
        if len(appliance) > 0:
            return appliance[0]
        else:
            handling = split_entry.replace('_' + appliance_name, '')
            return PredictModel.get_appliance(handling, '_' + appliance_name)