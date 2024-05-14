from core.Migration import Migration
from SQL.SQLQueries import PowerUsageOperations as Query

class M004PowerUsage(Migration):
    def up(self):
        self.add_sql(Query.CREATE_POWER_USAGE_TABLE)
        self.add_sql(Query.CREATE_POWER_USAGE_INDEX)

    def down(self):
        self.add_sql(Query.DROP_POWER_USAGE_TABLE)