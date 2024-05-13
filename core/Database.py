import os

import mysql.connector as mysql
from dotenv import dotenv_values


class Database:
    def __init__(self, create_db=False):
        # Get env variables
        if create_db:
            self.create_db()
        self.connection = Database.connection()

    def create_db(self):
        connection = mysql.connect(
            host=dotenv_values()['DB_HOST'],
            user=dotenv_values()['DB_USERNAME'],
            passwd=dotenv_values()['DB_PASSWORD']
        )
        connection.cursor().execute('CREATE DATABASE IF NOT EXISTS ' + dotenv_values()['DB_DATABASE'])
        connection.close()

    @staticmethod
    def connection():
        return mysql.connect(
            host=dotenv_values()['DB_HOST'],
            user=dotenv_values()['DB_USERNAME'],
            passwd=dotenv_values()['DB_PASSWORD'],
            database=dotenv_values()['DB_DATABASE']
        )

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

        query = 'SELECT * FROM ' + table_name
        if where:
            query += ' WHERE ' + ' AND '.join([f"{k} = %({k})s" for k in where.keys()])

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
