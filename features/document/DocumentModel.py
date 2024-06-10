import time
from datetime import datetime, timedelta
import math
from time import strptime

import numpy as np
import pandas as pd

import app
from core.Database import Database

from SQL.SQLQueries import DatabaseOperations as Query
from SQL.SQLQueries import PowerUsageApplianceOperations as PowerUsageApplianceQuery
from SQL.SQLQueries import IsUsingApplianceOperations as IsUsingApplianceQuery
from SQL.SQLQueries import PowerUsageOperations as PowerUsageQuery
from features.predict.PredictModel import PredictModel
from features.train.TrainModel import TrainModel
from helper.ResultDocumentationHelper import ResultDocumentationHelper


class DocumentModel:
    def __init__(self, model):
        self.model = model
        self.train_model = TrainModel(self.model)

    def get_interesting_datetime(self, appliance, timeframes_s: int = 60):

        power_usage_appliance = Database.query(PowerUsageApplianceQuery.SELECT_JOIN_POWER_USAGE_WHERE.format('Appliance_id', appliance['id']))

        power_usage_appliance_df = pd.DataFrame(power_usage_appliance)

        most_interesting_datetime = power_usage_appliance_df['datetime'][0]
        most_interesting_difference = 0

        # Convert the datetime strings to datetime objects
        power_usage_appliance_df['timestamp'] = pd.array([datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp() for x in power_usage_appliance_df['datetime']])

        upper_bound = power_usage_appliance_df['appliance_power'].max() * 0.8
        lower_bound = power_usage_appliance_df['appliance_power'].max() * 0.2

        # Get the power_usage values that are above the lower bound and below the upper bound
        interesting_power_usage_appliance_df = power_usage_appliance_df[(power_usage_appliance_df['appliance_power'] > lower_bound) & (power_usage_appliance_df['appliance_power'] < upper_bound)]

        for index, row in interesting_power_usage_appliance_df.iterrows():
            # Get the previous and next timeframes
            previous_timeframe = power_usage_appliance_df['timestamp'][index - (timeframes_s / 3)]
            next_timeframe = power_usage_appliance_df['timestamp'][index + (timeframes_s / 3)]

            # Calculate the difference between the previous and next timeframe
            difference = next_timeframe - previous_timeframe

            # If the difference is greater than the most interesting difference, update the most interesting difference and datetime
            if difference > most_interesting_difference:
                most_interesting_difference = difference
                most_interesting_datetime = row['datetime']

        next_datetime = datetime.strptime(most_interesting_datetime, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=timeframes_s)

        power_usage = Database.query(PowerUsageQuery.SELECT_POWER_USAGE_BETWEEN.format(most_interesting_datetime, next_datetime))

        power_usage_df = pd.DataFrame(power_usage)

        appliance_in_use = []
        for power_usage_i in power_usage:
            appliance_in_use.append(bool(Database.query(
                IsUsingApplianceQuery.SELECT_WHERE_POWERUSAGE_FOR_APPLIANCE.format(power_usage_i['id'], appliance['id']))))

        return power_usage_df['datetime'].tolist(), power_usage_df['power_usage'].tolist(), appliance_in_use

    def train(self, appliance, epochs, print_progress: bool = True):
        return self.train_model.train(appliance, epochs, print_progress)


    def predict(self, model_path, appliance, datetime, power_usage, appliance_in_use, print_progress: bool = True, show_plot: bool = True):
        predict_model = PredictModel(model_path, appliance)
        predictions, probabilities = predict_model.predict(datetime, power_usage, appliance_in_use, print_progress)
        return predict_model.visualize(predictions, appliance_in_use, probabilities, show_plot)

    def generate_document(self, image_path=None, chosen_datetime=None, appliance=None, custom_parameters: dict = None):
        document_helper = ResultDocumentationHelper(self.model.__class__.__name__)
        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        document_helper.add_heading(f"[{datetime_str}] Model prediction", level=1)

        if appliance is not None:
            document_helper.add_parameter('Appliance', appliance)
        if chosen_datetime is not None:
            document_helper.add_parameter('Chosen datetime', chosen_datetime)

        if self.model.get_document_parameters() is not None:
            for key, value in self.model.get_document_parameters().items():
                document_helper.add_parameter(key, value)

        if custom_parameters is not None:
            for key, value in custom_parameters.items():
                document_helper.add_parameter(key, value)

        if image_path is not None:
            document_helper.add_image(image_path)

        document_helper.add_break()

        document_helper.save()
