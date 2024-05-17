from core.Generate import Generate
from datetime import datetime, time
from features.generate.synthetic.SyntheticBuilding import SyntheticBuilding
from core.Database import Database
from datetime import timedelta
import random

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