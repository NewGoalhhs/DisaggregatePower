import logging

from helper.LoadingBarHelper import LoadingBarHelper


class Migration:
    def __init__(self, db):
        self.db = db
        self.queries = []
        self.lb = None

        # Set up logging
        logging.basicConfig(filename='migration.log', level=logging.INFO)
        self.logger = logging.getLogger()

    def add_loading_bar(self, lb):
        self.lb = lb

    def update_loading_bar(self, current):
        if self.lb is not None:
            self.lb.update(current)

    def finish_loading_bar(self):
        if self.lb is not None:
            self.lb.finish()

    def set_loading_bar_goal(self, goal):
        if self.lb is not None:
            self.lb.set_goal(goal)

    def set_loading_bar_status(self, status):
        if self.lb is not None:
            self.lb.set_status(status)

    def get_cursor(self):
        return self.db.connection.cursor()

    def add_sql(self, sql):
        self.queries.append(sql)

    def migrate(self):
        cursor = self.get_cursor()
        self.set_loading_bar_status('Running queries')
        for query in self.queries:
            self.update_loading_bar(1)
            try:
                cursor.execute(query)
            except Exception as e:
                self.logger.error('Query failed: ' + query + ' with error: ' + str(e))
        self.db.connection.commit()
        cursor.close()
        self.finish_loading_bar()
        self.reset_queries()

    def reset_queries(self):
        self.queries = []

    def up(self):
        pass

    def down(self):
        pass

    def insert(self, csv_path):
        pass