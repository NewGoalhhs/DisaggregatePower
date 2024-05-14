from core.Migration import Migration
from SQL.SQLQueries import ApplianceOperations as Query

class M003Appliance(Migration):
    def up(self):
        self.add_sql(Query.CREATE_APPLIANCE_TABLE)

    def down(self):
        self.add_sql(Query.DROP_APPLIANCE_TABLE)