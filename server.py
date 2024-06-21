from flask import Flask
from flask_cors import CORS

from api.Api import Api
from flask_app import app, socketio

# Example data that your application might use
data = {
    "example_key": "example_value"
}

api = Api(app, socketio)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/socket.io/*": {"origins": "*"}})

if __name__ == '__main__':
    api.run()
