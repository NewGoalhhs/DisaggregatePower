from core.Generate import Generate
from helper.PrintHelper import PrintHelper
from model.Appliance import Appliance
from datetime import datetime, time
from features.generate.synthetic.SyntheticBuilding import SyntheticBuilding
from core.Database import Database
from SQL.SQLQueries import ApplianceOperations as ApplianceQuery

class SimpleGenerate(Generate):
    def __init__(self):
        super().__init__()
        self.db = Database()

    def run(self, p):
        # Generate a new building
        building = SyntheticBuilding()

        start_time = datetime.strftime("2011-02-01 00:00:00")
        end_time = datetime.strftime("2011-05-01 00:00:00")
        print(building.id, start_time, end_time)
        # Generate a random number of appliances with a base count of half the amount of appliances
        # appliances = Database.query(ApplianceQuery)
        # Get random distinct appliance types from the database

        # Copy the appliance power usage from a random building and set the datetime to the random timeframe

        #
        pass
