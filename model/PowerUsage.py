from abc import ABC

from core.DbModel import DbModel


class PowerUsageAppliance(DbModel, ABC):
    def __init__(self):
        self.id = None
        self.building_id = None
        self.datetime = None
        self.power_usage = None

    def save(self):
        pass

    def delete(self):
        pass

    @staticmethod
    def table_name() -> str:
        return 'PowerUsage'
