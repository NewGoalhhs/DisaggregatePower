from core.Migration import Migration
from SQL.SQLQueries import MigrationOperations as Query


class M001Migrations(Migration):
    def up(self):
        self.add_sql(Query.CREATE_MIGRATIONS_TABLE)

    def down(self):
        self.add_sql(Query.DROP_MIGRATIONS_TABLE)
