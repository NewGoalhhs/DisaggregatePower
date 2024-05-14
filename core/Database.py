from dotenv import dotenv_values
import sqlite3
from SQL.SQLQueries import DatabaseOperations as Query

class Database:
    def __init__(self, create_db=False):
        # Get env variables
        if create_db:
            self.create_db()
        self.connection = Database.connection()

    def create_db(self):
        connection = sqlite3.connect(dotenv_values()['DB_HOST'])
        connection.cursor().execute(Query.CREATE_DB.format(dotenv_values()['DB_DATABASE']))
        connection.close()

    @staticmethod
    def connection():
        return sqlite3.connect(dotenv_values()['DB_HOST'] + dotenv_values()['DB_DATABASE'])

    @staticmethod
    def query(query):
        connection = Database.connection()
        cursor = connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        connection.commit()
        return results

    @staticmethod
    def fetch_with(model, table_name, where=None):
        connection = Database.connection()
        cursor = connection.cursor(dictionary=True)

        query = Query.SELECT_ALL.format(table_name)
        if where:
            query = Query.SELECT_WHERE.format(table_name, ' AND '.join([f"{k} = %({k})s" for k in where.keys()]))

        print(query)
        cursor.execute(query, where)
        results = []
        for result in cursor.fetchall():
            model_instance = model()
            for key, value in result.items():
                setattr(model_instance, key, value)
            results.append(model_instance)
        connection.commit()
        return results

    @staticmethod
    def get_next_id(table_name):
        connection = Database.connection()
        cursor = connection.cursor()
        cursor.execute(Query.SELECT_MAX.format('id', table_name))
        result = cursor.fetchone()
        connection.commit()
        return result[0] + 1 if result[0] else 1