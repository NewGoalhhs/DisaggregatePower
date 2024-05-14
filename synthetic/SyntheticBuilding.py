from core.Database import Database
from model.Building import Building

class SyntheticBuilding:
    def __init__(self):
        self.id = Database.get_next_id(Building.table_name())
        self.name = 'SYNTH_' + str(self.id)

    def save(self):
        Database.query(f'INSERT INTO {Building.table_name()} (id, name) VALUES ({self.id}, "{self.name}")')

