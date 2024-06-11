from core.Database import Database
from SQL.SQLQueries import PowerUsageApplianceOperations as Query


class SyntheticPowerUsage_Appliance:
    def __init__(self, power_usage_id, appliance_id, appliance_power):
        self.power_usage_id = power_usage_id
        self.appliance_id = appliance_id
        self.appliance_power = appliance_power

    def save(self):
        Database.query(Query.INSERT_POWER_USAGE_APPLIANCE.format(self.power_usage_id, self.appliance_id, self.appliance_power))