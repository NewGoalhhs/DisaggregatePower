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
        appliance_id = self.prepare_train(p)
        lb = p.get_loading_bar(text="Training model", goal=5)
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
        data = {
            "timestamp": power_usage['datetime'],
            "power_usage": power_usage['power_usage'],
            "appliance_in_use": power_usage['IsUsingAppliance']
        }
        lb.finish()
        #
        # print(len(data['timestamp']), len(data['power_usage']), len(data['appliance_in_use']))
        #
        self.model.train(data)
        self.model.save_model(self.get_save_path(appliance_id))

    def prepare_train(self, p):
        appliances = Database.query(Query.SELECT_ALL.format('Appliance'))
        p.print_line("All appliances:")
        p.open_options()
        for index, appliance in enumerate(appliances):
            p.add_option(index+1, appliance['name'], appliance['id'])
        appliance_id = p.choose_option('Choose an appliance to train: ')
        return appliance_id

    def get_save_path(self, appliance_id):
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        return app.__ROOT__ + f"/MachineLearning/models/{self.model.__name__}/{datetime_str}_{appliance_id}.pkl"
