import os

import mysql.connector as mysql
from dotenv import dotenv_values


class Database:
    def __init__(self, create_db=False):
        # Get env variables
        if create_db:
            self.create_db()
        env = dotenv_values()
        self.connection = mysql.connect(
            host=env['DB_HOST'],
            user=env['DB_USERNAME'],
            passwd=env['DB_PASSWORD'],
            database=env['DB_DATABASE']
        )

    def create_db(self):
        connection = mysql.connect(
            host=dotenv_values()['DB_HOST'],
            user=dotenv_values()['DB_USERNAME'],
            passwd=dotenv_values()['DB_PASSWORD']
        )
        connection.execute('CREATE DATABASE IF NOT EXISTS ' + dotenv_values()['DB_DATABASE'])
        connection.close()
