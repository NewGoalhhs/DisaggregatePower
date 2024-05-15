import logging

class Migration:
    def __init__(self, db):
        self.db = db
        self.queries = []

        # Set up logging
        logging.basicConfig(filename='migration.log', level=logging.INFO)
        self.logger = logging.getLogger()

    def get_cursor(self):
        return self.db.connection.cursor()

    def add_sql(self, sql):
        self.queries.append(sql)

    def migrate(self):
        cursor = self.get_cursor()
        for query in self.queries:
            try:
                cursor.execute(query)
            except Exception as e:
                self.logger.error('Query failed: ' + query + ' with error: ' + str(e))
        self.db.connection.commit()
        cursor.close()
        self.reset_queries()

    def reset_queries(self):
        self.queries = []

    def up(self):
        pass

    def down(self):
        pass

    def insert(self, csv_path):
        pass