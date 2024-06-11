from flask import jsonify, request

from SQL.SQLQueries import DatabaseOperations as Query
from core.Database import Database
from flask_app import app


@app.route('/api/appliance', methods=['GET', 'POST'])
def get_appliances():
    if request.method == 'GET':
        appliances = Database.query(Query.SELECT_ALL.format('Appliance'))
        return jsonify(appliances)
    else:
        pass


@app.route('/api/appliance/<int:id>', methods=['GET', 'POST'])
def get_appliance(id: int):
    if request.method == 'GET':
        appliance = Database.query(Query.SELECT_WHERE.format('Appliance', 'id', id))
        return jsonify(appliance)
    else:
        pass
