from core.Migration import Migration
from SQL.SQLQueries import IsUsingApplianceOperations as Query
from core.Database import Database
from SQL.SQLQueries import DatabaseOperations as DatabaseQuery
import pandas as pd

class IsUsingAppliance(Migration):

    def up(self):
        self.add_sql(Query.CREATE_IS_USING_APPLIANCE_TABLE)

    def down(self):
        self.add_sql(Query.DROP_IS_USING_APPLIANCE_TABLE)

    def insert(self):
        self.set_loading_bar_status('Preparing data')
        self.update_loading_bar(0)


        appliance_thresholds = {}
        for id, name in Database.query(DatabaseQuery.SELECT_ALL.format('Appliance')):
            appliance_thresholds[id] = Database.query(DatabaseQuery.SELECT_APPLIANCE_THRESHOLD.format(id))[0][0]

        # get all building ids
        building_ids = Database.query(DatabaseQuery.SELECT_ALL.format('Building'))

        for building_id in building_ids:
            building_id = building_id[0]
            power_usage_appliance = Database.query(DatabaseQuery.SELECT_POWER_USAGE_APPLIANCE_FOR_BUILDING.format(building_id))
            print(len(power_usage_appliance))
            for power_usage_id, appliance_id, appliance_power in power_usage_appliance:
                if appliance_power > appliance_thresholds[appliance_id]:
                    # self.add_sql(Query.INSERT_IS_USING_APPLIANCE.format(power_usage_id, appliance_id, appliance_power))
                    pass
                self.update_loading_bar(1)


