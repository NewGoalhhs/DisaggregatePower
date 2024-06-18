from flask import Flask

from api.Router import Router
from core.Database import Database


class Api:
    def __init__(self, app):
        self.app = app
        self.db = Database()
        self.router = Router()

    def run(self):
        self.router.init_routes()
        self.app.run(host='0.0.0.0', port=4201)
