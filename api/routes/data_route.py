from flask_app import app


@app.route('/api/train_options', methods=['GET', 'POST'])
def get_data():
    models =


    if request.method == 'GET':
        return jsonify(data)
    else:
        pass