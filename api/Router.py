import os

from app import __ROOT__
from flask_app import socketio


class Router:
    def init_routes(self):
        for route in self.get_routes():
            __import__(route.replace('/', '.'), fromlist=[''])

    def get_routes(self, path='api/routes', max_depth=5):
        if max_depth == 0:
            return

        routes = []

        with os.scandir(__ROOT__ + '/' + path) as entries:
            for entry in entries:
                new_path = path + '/' + entry.name
                if entry.is_dir():
                    routes += self.get_routes(new_path, max_depth - 1)
                else:
                    if entry.name.endswith('.py'):
                        routes.append(new_path.replace('.py', ''))

        return routes
