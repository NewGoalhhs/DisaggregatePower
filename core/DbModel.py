from core.Database import Database


class DbModel:
    def save(self):
        pass

    def delete(self):
        pass

    def get(self, key: str):
        return getattr(self, key)

    def set(self, key: str, value):
        setattr(self, key, value)

    @staticmethod
    def table_name() -> str:
        pass

    @classmethod
    def  fetch_with(cls, where: dict) -> list:
        result = Database.fetch_with(cls, cls.table_name(), where)
        return result

    @classmethod
    def fetch_all(cls) -> list:
        return cls.fetch_with({})
