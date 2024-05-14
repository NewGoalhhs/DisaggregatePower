from core.Migration import Migration
from SQL.SQLQueries import PowerUsageApplianceOperations as Query

class M005PowerUsageAppliance(Migration):
    def up(self):
        self.add_sql(Query.CREATE_POWER_USAGE_APPLIANCE_TABLE)
        self.add_sql(Query.CREATE_POWER_USAGE_APPLIANCE_INDEX)

    def down(self):
        self.add_sql(Query.DROP_POWER_USAGE_APPLIANCE_TABLE)