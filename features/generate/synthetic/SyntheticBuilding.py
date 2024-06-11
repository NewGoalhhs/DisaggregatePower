from core.Database import Database
from model.Building import Building
from SQL.SQLQueries import BuildingOperations as Query
from SQL.SQLQueries import DatabaseOperations as DbQuery

class SyntheticBuilding:
    def __init__(self):
        id = Database.get_next_id(Building.table_name())
        self.id = id
        self.name = 'SYNTH_' + str(id)

    def save(self):
        result = Database.query(Query.INSERT_BUILDING.format(self.name))
        self.id = result[0].get('id')