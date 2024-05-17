from core.Generate import Generate
from datetime import datetime
from features.generate.synthetic.SyntheticBuilding import SyntheticBuilding
from core.Database import Database
from datetime import timedelta
import random
from SQL.SQLQueries import DatabaseOperations as Query

class SimpleGenerate(Generate):
    def __init__(self):
        super().__init__()
        self.db = Database()

    def run(self, p):
        building = SyntheticBuilding()
        # building.save()

        start_time = datetime(2011, 2, 1, 0, 0, 0)

        end_time = datetime(2011, 6, 30, 23, 59, 59)

        current_time = start_time

        while current_time <= end_time:
            current_time = current_time + timedelta(seconds=random.randint(3, 10))
            # TODO: Maak eerst power data voor appliances en daarna main power. Main power zou alle appliances moeten bevatten + een beetje extra.
            for appliance in Database.query(Query.SELECT_ALL.format('appliance')):
                # {'id': 1, 'name': 'bathroom_gfi'}
                # Expected usage:
                # Weekday: 00:00 - 06:00: 0.1
                # Weekday: 06:00 - 12:00: 0.2
                # Weekday: 12:00 - 18:00: 0.3
                # Weekday: 18:00 - 00:00: 0.4

                # Weekend: 00:00 - 06:00: 0.1
                # Weekend: 06:00 - 12:00: 0.2
                # Weekend: 12:00 - 18:00: 0.3
                # Weekend: 18:00 - 00:00: 0.4

                #TODO: Create a dictionary: {1: {weekday: {00:00-00:30: 0.1, 00:30-01:00: 0.2, etc...}}, 2: {etc...}, etc...}

                # {'id': 2, 'name': 'dishwaser'}
                # {'id': 3, 'name': 'electric_heat'}
                # {'id': 4, 'name': 'kitchen_outlets'}
                # {'id': 5, 'name': 'lighting'}
                # {'id': 6, 'name': 'microwave'}
                # {'id': 7, 'name': 'oven'}
                # {'id': 8, 'name': 'refrigerator'}
                # {'id': 9, 'name': 'stove'}
                # {'id': 10, 'name': 'washer_dryer'}
                # {'id': 11, 'name': 'disposal'}
                # {'id': 12, 'name': 'electronics'}
                # {'id': 13, 'name': 'furance'}
                # {'id': 14, 'name': 'outlets_unknown'}
                # {'id': 15, 'name': 'smoke_alarms'}
                # {'id': 16, 'name': 'air_conditioning'}
                # {'id': 17, 'name': 'miscellaeneous'}
                # {'id': 18, 'name': 'outdoor_outlets'}
                # {'id': 19, 'name': 'subpanel'}
                pass