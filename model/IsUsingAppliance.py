from abc import ABC

from core.DbModel import DbModel


class IsUsingAppliance(DbModel, ABC):
    def __init__(self):
        self.id = None
        self.power_usage_id = None
        self.appliance_id = None

    def save(self):
        pass

    def delete(self):
        pass

    @staticmethod
    def table_name() -> str:
        return 'IsUsingAppliance'