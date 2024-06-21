from flask import Flask

from api.Router import Router
from api.routes.home_route import background_home_content_task_advanced
from core.Database import Database
import threading

class Api:
    def __init__(self, app, socket):
        self.app = app
        self.socket = socket
        self.db = Database()
        self.router = Router()

    def run(self):
        self.router.init_routes()
        self.app.run(host='0.0.0.0', port=4201)
        self.socket.run(self.app)
