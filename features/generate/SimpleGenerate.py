import pandas as pd

from core.Generate import Generate
from datetime import datetime
from features.generate.synthetic.SyntheticBuilding import SyntheticBuilding
from core.Database import Database
from datetime import timedelta
import random
from SQL.SQLQueries import DatabaseOperations as Query
from SQL.SQLQueries import PowerUsageApplianceOperations as PowerUsageApplianceQuery
from helper.LoadingBarHelper import LoadingBarHelper


class SimpleGenerate(Generate):
    def __init__(self):
        super().__init__()
        self.db = Database()
        self.lb = None

    def run(self, p):
        # self.lb = LoadingBarHelper('Synthetic data - ', 100, 0)
        building = SyntheticBuilding()
        # building.save()

        start_time = datetime(2011, 2, 1, 0, 0, 0)

        end_time = datetime(2011, 6, 30, 23, 59, 59)

        current_time = start_time

        # Get random amount of appliances to generate data for between half and the full amount of appliances
        appliances = self.db.query(Query.SELECT_ALL.format('appliance'))
        appliances = random.sample(appliances, random.randint(int(len(appliances) / 2), len(appliances)))

        # Predict the amount of operations required for the loading bar
        # self.lb.set_goal(int((end_time - start_time).total_seconds()) * len(appliances))

        # {'id': 1, 'name': 'bathroom_gfi'}
        appliance_power_usages = {
            1: {
                'weekday': {
                    '00:00-00:30': 0.0-0.1,
                },
                'weekend': {

                }
            },
            2: {

            }
        }

        for appliance in appliances:
            powerUsage = self.get_power_usage_appliance_from_appliance(appliance['id'])
            # Create dataframe
            df = pd.DataFrame(powerUsage)
            # Print the dataframe avg, mean, min, max, 25%,etc
            print(df.describe())

        # while current_time < end_time:
        #     random_time_gap = random.randint(3, 10)
        #     current_time = current_time + timedelta(seconds=random_time_gap)
        #     # TODO: Maak eerst power data voor appliances en daarna main power. Main power zou alle appliances moeten bevatten + een beetje extra.
        #
        #     power_usage_appliances = []
        #     self.lb.set_status('Gathering data')
        #     for appliance in appliances:
        #         # power_usages = self.get_random_power_usage_appliance_from_appliance(appliance['id'])
        #         self.lb.update(random_time_gap)
        #         # Generate power data for appliance
        #         # {'id': 1, 'name': 'bathroom_gfi'}
        #         # Expected usage:
        #         # Weekday: 00:00 - 06:00: 0.1-0.2
        #         # Weekday: 06:00 - 12:00: 0.2-0.7
        #         # Weekday: 12:00 - 18:00: 0.3
        #         # Weekday: 18:00 - 00:00: 0.4
        #
        #         # Weekend: 00:00 - 06:00: 0.1
        #         # Weekend: 06:00 - 12:00: 0.2
        #         # Weekend: 12:00 - 18:00: 0.3
        #         # Weekend: 18:00 - 00:00: 0.4
        #
        #
        #
        #         #TODO: Create a dictionary: {1: {weekday: {00:00-00:30: 0.1, 00:30-01:00: 0.2, etc...}}, 2: {etc...}, etc...}
        #
        #         # {'id': 2, 'name': 'dishwaser'}
        #         # {'id': 3, 'name': 'electric_heat'}
        #         # {'id': 4, 'name': 'kitchen_outlets'}
        #         # {'id': 5, 'name': 'lighting'}
        #         # {'id': 6, 'name': 'microwave'}
        #         # {'id': 7, 'name': 'oven'}
        #         # {'id': 8, 'name': 'refrigerator'}
        #         # {'id': 9, 'name': 'stove'}
        #         # {'id': 10, 'name': 'washer_dryer'}
        #         # {'id': 11, 'name': 'disposal'}
        #         # {'id': 12, 'name': 'electronics'}
        #         # {'id': 13, 'name': 'furance'}
        #         # {'id': 14, 'name': 'outlets_unknown'}
        #         # {'id': 15, 'name': 'smoke_alarms'}
        #         # {'id': 16, 'name': 'air_conditioning'}
        #         # {'id': 17, 'name': 'miscellaeneous'}
        #         # {'id': 18, 'name': 'outdoor_outlets'}
        #         # {'id': 19, 'name': 'subpanel'}

        # self.lb.finish()

    def get_random_power_usage_appliance_from_appliance(self, appliance_id):
        building_id = self.get_random_building_id_with_appliance(appliance_id)
        if building_id:
            power_usages = self.db.query(Query.SELECT_POWER_USAGE_APPLIANCE_FOR_BUILDING.format(building_id))
            return power_usages
        return []

    def get_random_building_id_with_appliance(self, appliance_id):
        result = self.db.query(Query.SELECT_BUILDING_IDS_WITH_APPLIANCE.format(appliance_id))
        return random.choice(result)['building_id']

    def insert_power_usage_appliance(self, power_usages):
        connection = self.db.connection()
        connection.fast_executemany = True
        with connection.cursor() as cursor:
            cursor.executemany(PowerUsageApplianceQuery.INSERT_POWER_USAGE_APPLIANCE, power_usages)
        connection.commit()

    def get_power_usage_appliance_from_appliance(self, appliance_id):
        return self.db.query(Query.SELECT_POWER_USAGE_APPLIANCE_FOR_APPLIANCE.format(appliance_id))
