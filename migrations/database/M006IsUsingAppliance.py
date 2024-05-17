# TODO: Implement a feature that checks if the user is using an appliance and defines that in the database.
from core.Migration import Migration
from SQL.SQLQueries import IsUsingApplianceOperations as Query

class IsUsingAppliance(Migration):
    def up(self):
        self.add_sql(Query.CREATE_IS_USING_APPLIANCE_TABLE)

    def down(self):
        self.add_sql(Query.DROP_IS_USING_APPLIANCE_TABLE)