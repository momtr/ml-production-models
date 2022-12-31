import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from io import BytesIO
import joblib
import uuid
from pymongo import MongoClient
from mappers import map_to_model
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/health')
@cross_origin()
def health_check():
    return 'success'


@app.route('/models/train', methods=['POST'])
@cross_origin()
def train_model():
    data = request.get_json()
    if 'type' not in data:
        return jsonify({ 'error': 'BAD_REQUEST', 'message': 'type not specified' }), 400

    if 'name' not in data:
        return jsonify({ 'error': 'BAD_REQUEST', 'message': 'name not specified' }), 400

    if 'data' not in data:
        return jsonify({ 'error': 'BAD_REQUEST', 'message': 'data not specified' }), 400

    if data['type'] == 'text-classification':

        if 'x_values' not in data['data']:
            return jsonify({ 'error': 'BAD_REQUEST', 'message': 'data.x_values not specified' }), 400

        if 'y_values' not in data['data']:
            return jsonify({ 'error': 'BAD_REQUEST', 'message': 'data.y_values not specified' }), 400

        # train model
        count_vec = CountVectorizer(stop_words='english')
        series = pd.Series(data['data']['x_values'])
        bow = count_vec.fit_transform(series)
        bow = np.array(bow.todense())
        X = bow
        Y = pd.Series(data['data']['y_values'])
        split_ratio = 0.5
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_ratio, stratify=Y)
        model = MultinomialNB()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        # serialize the model and the vectorizer
        bytes_container = BytesIO()
        joblib.dump(model, bytes_container)
        bytes_container.seek(0)
        bytes_model = bytes_container.read()

        bytes_container_vec = BytesIO()
        joblib.dump(count_vec, bytes_container_vec)
        bytes_container_vec.seek(0)
        bytes_vectorizer = bytes_container_vec.read()

        # measure performance metrics
        accuracy_score_value = accuracy_score(Y_test, Y_pred)
        f1_score_value = f1_score(Y_test, Y_pred, average="macro")

        # save the model in the databse
        client = MongoClient(get_mongodb_connection_string())
        db = client['demo']
        collection = db['models']
        record_uuid = str(uuid.uuid4())
        record = {
            "type": data['type'],
            "inputs": data['data'],
            "name": data['name'],
            "model": bytes_model,
            "count_vectorizer": bytes_vectorizer,
            "accuracy_score": accuracy_score_value,
            "split_ratio": split_ratio,
            "f1_score": f1_score_value,
            "uuid": record_uuid,
            "size": len(data['data']['x_values']),
            "active": True,
            "input_variables": [ "text" ],
            "output_variables": [ "class" ],
            "details": {
                "classes": Y.unique().tolist()
            }
        }
        x = collection.insert_one(record)

        new_record = collection.find_one({ "uuid": record_uuid })

        # send test results and model id back to client
        return jsonify(map_to_model(new_record)), 200

    else:
        return jsonify({ 'error': 'TYPE_DOES_NOT_EXIST', 'message': 'The specified model type does not exist' }), 404


@app.route('/models', methods=['GET'])
@cross_origin()
def get_all_models():
    
    client = MongoClient(get_mongodb_connection_string())
    db = client['demo']
    models = db['models']
    records = models.find({ "active": True })

    return jsonify(list(map(map_to_model, records))), 200


@app.route('/models/<model_uuid>', methods=['DELETE'])
@cross_origin()
def invalidate_model(model_uuid):

    client = MongoClient(get_mongodb_connection_string())
    db = client['demo']
    models = db['models']
    old_model = models.find_one({ "uuid": model_uuid })

    if old_model is None:
        return jsonify({ 'error': 'MODEL_NOT_FOUND', 'message': 'Model with specified id not found' }), 400

    records = models.update_one({ "active": True, "uuid": model_uuid }, { "$set": { "active": False } })

    return jsonify({ "status": "success" }), 200


@app.route('/models/predict', methods=['POST'])
@cross_origin()
def predict_with_model():
    data = request.get_json()

    if 'model_id' not in data:
        return jsonify({ 'error': 'BAD_REQUEST', 'message': 'model_id not specified' }), 400

    if 'text_to_classify' not in data:
        return jsonify({ 'error': 'BAD_REQUEST', 'message': 'text_to_classify not specified' }), 400
    
    # get record from DB
    client = MongoClient(get_mongodb_connection_string())
    db = client['demo']
    models = db['models']
    model = models.find_one({ "uuid": data['model_id'] })

    if model is None:
        return jsonify({ 'error': 'MODEL_NOT_FOUND', 'message': 'Model with specified id not found' }), 400

    if model['type'] == 'text-classification':

        # deserialize model and count vectorizer
        count_vectorizer = joblib.load(BytesIO(model['count_vectorizer']))
        model = joblib.load(BytesIO(model['model']))

        # make prediction
        bow_prediction = count_vectorizer.transform(pd.Series(data['text_to_classify']))
        bow_prediction = np.array(bow_prediction.todense())
        prediction = model.predict(bow_prediction)
        prediction_class = prediction[0]

        # store prediction in DB
        record = {
            "model_id": data['model_id'],
            "text_to_classify": data['text_to_classify'],
            "classification": str(prediction_class)
        }
        x = db['predictions'].insert_one(record)

        # send prediction to client
        return jsonify({
            "model_id": data['model_id'],
            "input_variables": {
                "text": data['text_to_classify']
            },
            "output_variables": {
                "class": str(prediction_class)
            },
            "insert_id": str(x.inserted_id)
        }), 200

    else:
        return jsonify({ 'error': 'NOT_IMPLEMENTED', 'message': 'Preditions for the specified model type do not exist' }), 500


def get_mongodb_connection_string():
    con_str = os.environ.get('MONGO_CONNECTION_STRING')
    if con_str is not None:
        return con_str
    else:
        return "mongodb://127.0.0.1:27017"

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
