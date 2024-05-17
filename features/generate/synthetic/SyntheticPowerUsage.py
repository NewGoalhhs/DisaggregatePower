from core.Database import Database
from SQL.SQLQueries import PowerUsageOperations as Query


class SyntheticPowerUsage:
    def __init__(self, building_id, datetime, power_usage):
        self.building_id = building_id
        self.datetime = datetime
        self.power_usage = power_usage

    def save(self):
        Database.query(Query.INSERT_POWER_USAGE.format(self.building_id, self.datetime, self.power_usage))

