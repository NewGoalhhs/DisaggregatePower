from core.DbModel import DbModel


class PowerUsageAppliance(DbModel):
    def __init__(self):
        self.id = None
        self.name = None

    def __init__(self, id, name):
        if not isinstance(id, int):
            raise ValueError("id must be an integer")
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        self.id = id
        self.name = name

    def save(self):
        pass

    def delete(self):
        pass

    @staticmethod
    def table_name() -> str:
        return 'Building'