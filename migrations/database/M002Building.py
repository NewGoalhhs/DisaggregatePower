from core.Migration import Migration
from SQL.SQLQueries import BuildingOperations as Query


class M002Building(Migration):
    def up(self):
        self.add_sql(Query.CREATE_BUILDING_TABLE)

    def down(self):
        self.add_sql(Query.DROP_BUILDING_TABLE)

    def insert(self, csv_path):
        self.set_loading_bar_goal(2)
        name = ("REDDUS_" + csv_path.split('_')[1].split('.')[0])
        self.update_loading_bar(1)
        self.add_sql(Query.INSERT_BUILDING.format(name))


