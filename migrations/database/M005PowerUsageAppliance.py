import pandas as pd

from core.Database import Database
from core.Migration import Migration
from SQL.SQLQueries import PowerUsageApplianceOperations as Query
from SQL.SQLQueries import BuildingOperations as BuildingQuery
from SQL.SQLQueries import DatabaseOperations as DatabaseQuery


class M005PowerUsageAppliance(Migration):
    def up(self):
        self.add_sql(Query.CREATE_POWER_USAGE_APPLIANCE_TABLE)
        self.add_sql(Query.CREATE_POWER_USAGE_APPLIANCE_INDEX)

    def down(self):
        self.add_sql(Query.DROP_POWER_USAGE_APPLIANCE_TABLE)

    def insert(self, csv_path):
        power_usage_lookup = set()
        self.set_loading_bar_status('Preparing data')
        self.update_loading_bar(0)

        building = ("REDDUS_" + csv_path.split('_')[1].split('.')[0])
        building_id = Database.query(BuildingQuery.GET_BUILDING_ID.format(building))[0][0]

        df = pd.read_csv(csv_path).dropna()
        df['time'] = pd.to_datetime(df['time'])


        for datetime in df['time'].unique():
            power_usage_lookup.add((building_id, datetime))

        # Create a dictionary to map (building_id, datetime) to PowerUsage id
        power_usage_ids = {}

        appliance_ids = {}

        for id, name in Database.query(DatabaseQuery.SELECT_ALL.format('Appliance')):
            appliance_ids[name] = id

        for id, b_id, datetime, main in Database.query(DatabaseQuery.SELECT_ALL.format('PowerUsage')):
            power_usage_ids[(b_id, datetime)] = id

        self.set_loading_bar_status('Retrieving data')
        self.set_loading_bar_goal(len(list(df.iterrows())) * len(df.columns) * 2)

        for index, row in df.iterrows():
            power_usage_id = power_usage_ids.get((building_id, str(row['time'])), None)

            if power_usage_id is not None:
                for header in df.columns:
                    if appliance_ids.get(header, None) is not None:
                        self.add_sql(Query.INSERT_POWER_USAGE_APPLIANCE.format(power_usage_id, appliance_ids[header], row[header]))
                    self.update_loading_bar(1)

