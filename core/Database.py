import os

import mysql.connector as mysql


class Database:
    def __init__(self):
        # Get env variables
        self.connection = mysql.connect(
            host=os.getenv('DB_HOST')+':'+os.getenv('DB_PORT'),
            user=os.getenv('DB_USER'),
            passwd=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_DATABASE')
        )
