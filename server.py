from flask import Flask
from flask_cors import CORS

from api.Api import Api
from flask_app import app

# Example data that your application might use
data = {
    "example_key": "example_value"
}

api = Api(app)
CORS(app, resources={r"/api/*": {"origins": "*"}})

if __name__ == '__main__':
    api.run()