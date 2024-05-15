from abc import ABC

from core.DbModel import DbModel


class Appliance(DbModel, ABC):
    def __init__(self):
        self.id = None
        self.name = None

    def save(self):
        pass

    def delete(self):
        pass

    @staticmethod
    def table_name() -> str:
        return 'Appliance'
