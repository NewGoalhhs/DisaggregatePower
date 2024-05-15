from core.Database import Database
from model.Building import Building
from SQL.SQLQueries import BuildingOperations as Query

class SyntheticBuilding:
    def __init__(self):
        id = Database.get_next_id(Building.table_name())
        self.name = 'SYNTH_' + str(id)

    def save(self):
        Database.query(Query.INSERT_BUILDING.format(self.name))

