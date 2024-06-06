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
from helper.ResultDocumentationHelper import ResultDocumentationHelper


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

    def prepare_predict(self, p):
        return self.use_existing_data(p)

    def run(self, p):
        self.predict(p)

        p.request_input("Press enter to continue: ")

        p.to_previous_screen()

    def predict(self, datetime, power_usage, appliance_in_use, print_progress: bool = True):
        p = PrintHelper()
        data = {
            "datetime": datetime,
            "power_usage": power_usage,
            "appliance_in_use": appliance_in_use
        }
        predictions, probabilities = self.model.predict(data)
        if print_progress:
            print(f"Datetime: {datetime[0]} - {datetime[-1]}")
            print(f"Power usage: {power_usage}")
            print(f"Predicted appliance usage: {predictions}")
            for power_usage_i, prediction in zip(power_usage, predictions):
                print("Prediction: " + str(power_usage_i) + ' - ' + str(prediction))

        return predictions, probabilities



    def get_datetime(self, p):
        current = datetime.datetime.now()
        return current.year, current.month, current.day, current.hour, current.minute, current.second

    def use_existing_data(self, p):
        # Get a random timeframe of 1 minute
        power_usage = Database.query(Query.SELECT_ALL.format('PowerUsage'))
        # Get a random datetime from power_usage
        # random_datetime = random.Random().choice(power_usage)['datetime']
        random_datetime = p.request_input('Enter a datetime: ')

        if random_datetime == '':
            return self.ask_user_for_datetime_and_power_usage(p)
        # Get the next minute from the random datetime
        next_minute = datetime.datetime.strptime(random_datetime, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(minutes=1)
        # Get the power usage from the random datetime to the next minute
        power_usage = Database.query(PowerUsageQuery.SELECT_POWER_USAGE_BETWEEN.format(random_datetime, next_minute))
        if len(power_usage) == 0:
            p.print_line('No power usage data available')
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

    def ask_user_for_datetime_and_power_usage(self, p):
        year, month, day, hour, minute, second = self.get_datetime(p)

        while True:
            input = p.request_input('Enter a day of the week and hour (1-7 0-23) [now]: ')
            if input == '':
                break
            split_input = input.split(' ')
            # year = int(split_input[0])
            # month = int(split_input[1])
            day = int(split_input[0])
            hour = int(split_input[1])
            if len(split_input) == 2 and 0 < int(split_input[0]) < 8 and 0 < int(split_input[1]) < 24:
                break
            # minute = int(split_input[4])
            # second = int(split_input[5])

            if len(split_input) == 6 and year > 0 and 0 < month < 12 and 0 < day < 31 and 0 < hour < 24 and 0 < minute < 60 and 0 < second < 60:
                break
            else:
                p.print_line('Please enter valid values')

        power_usage = 0

        while True:
            input = p.request_input('Enter the power usage amount (W): ')
            try:
                power_usage = int(input)
                break
            except ValueError:
                p.print_line('Please enter a valid power usage amount')

        datetime = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"

        return [datetime], [power_usage]

    def visualize(self, predictions, real_data, propabilities, show_plot=True):
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

        image_path = self.get_image_path()
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path)
        if show_plot:
            plt.show()
        return image_path

    def get_image_path(self):
        datetime_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return app.__ROOT__ + f"/src/{self.model_class}/{datetime_str}_{self.appliance['name']}.png"


