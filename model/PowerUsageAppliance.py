from abc import ABC

from core.DbModel import DbModel


class PowerUsageAppliance(DbModel, ABC):
    def __init__(self):
        self.id = None
        self.powerusage_id = None
        self.appliance_id = None
        self.appliance_power = None

    def save(self):
        pass

    def delete(self):
        pass

    @staticmethod
    def table_name() -> str:
        return 'PowerUsage_Appliance'
