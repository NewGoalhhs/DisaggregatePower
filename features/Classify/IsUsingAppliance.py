from core.Migration import Migration
from SQL.SQLQueries import IsUsingApplianceOperations as Query
from core.Database import Database
from SQL.SQLQueries import DatabaseOperations as DatabaseQuery
from helper.LoadingBarHelper import LoadingBarHelper


class IsUsingAppliance(Migration):

    def up(self):
        self.add_sql(Query.CREATE_IS_USING_APPLIANCE_TABLE)

    def down(self):
        self.add_sql(Query.DROP_IS_USING_APPLIANCE_TABLE)

    def insert(self):
        appliance_thresholds = {}
        for appliance in Database.query(DatabaseQuery.SELECT_ALL.format('Appliance')):
            try:
                appliance_power = Database.query(DatabaseQuery.SELECT_APPLIANCE_THRESHOLD.format(appliance.get('id')))[0].get('appliance_power')
                if isinstance(appliance_power, str):
                    appliance_power = appliance_power.replace(',', '.')
                appliance_thresholds[appliance.get('id')] = float(appliance_power)
            except IndexError:
                appliance_thresholds[appliance.get('id')] = 0
        # get all building ids
        building_ids = Database.query(DatabaseQuery.SELECT_ALL.format('Building'))

        for building_id in building_ids:
            lb = LoadingBarHelper('Classifying appliances for building: ' + building_id.get('name'), 1, 0)
            building_id = building_id.get('id')
            lb.set_status('Retrieving data')
            power_usage_appliance = Database.query(
                DatabaseQuery.SELECT_POWER_USAGE_APPLIANCE_FOR_BUILDING.format(building_id))
            lb.set_goal(len(power_usage_appliance))
            lb.set_status('Classifying appliances')
            for power_usage in power_usage_appliance:
                appliance_power = power_usage.get('appliance_power')
                if isinstance(appliance_power, str):
                    appliance_power = float(appliance_power.replace(',', '.'))
                if appliance_power > float(appliance_thresholds[power_usage.get('Appliance_id')]):
                    self.add_sql(Query.INSERT_IS_USING_APPLIANCE.format(power_usage.get('PowerUsage_id'),
                                                                        power_usage.get('Appliance_id')))
                lb.update(1)
            lb.finish()
