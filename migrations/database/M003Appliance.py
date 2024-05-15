import pandas as pd

from core.Migration import Migration
from SQL.SQLQueries import ApplianceOperations as Query


class M003Appliance(Migration):
    def up(self):
        self.add_sql(Query.CREATE_APPLIANCE_TABLE)

    def down(self):
        self.add_sql(Query.DROP_APPLIANCE_TABLE)

    def exclude_columns(self):
        return [
            'main',
            'time'
        ]

    def insert(self, csv_path):
        data = pd.read_csv(csv_path, nrows=1).columns
        self.set_loading_bar_goal(len(data) * 2)
        self.set_loading_bar_status('Retrieving data')
        for name in data:
            self.update_loading_bar(1)
            if name in self.exclude_columns():
                continue
            self.add_sql(Query.INSERT_APPLIANCE.format(name))
