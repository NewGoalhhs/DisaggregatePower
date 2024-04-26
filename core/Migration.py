
class Migration:
    def __init__(self, db):
        self.db = db
        self.queries = []

    def get_cursor(self):
        return self.db.connection.cursor()

    def add_sql(self, sql):
        self.queries.append(sql)

    def migrate(self):
        cursor = self.get_cursor()
        for query in self.queries:
            cursor.execute(query)
        self.db.connection.commit()
        cursor.close()
        self.reset_queries()

    def reset_queries(self):
        self.queries = []

    def up(self):
        pass

    def down(self):
        pass