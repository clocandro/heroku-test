from flask import Flask
import json
import pandas as pd
import pickle

from flask import request, jsonify

app = Flask(__name__)

#### Model preparation
model_file_name = "model.pkl"


def getModel():
    with open(model_file_name, 'rb') as pickled:
        model = pickle.load(pickled);
    return model;


model = getModel();

##end model preparation


@app.route('/')
def home():
    return 'I am listening!'


@app.route('/predict', methods=["GET", "POST"])
def predict():
    encoded_auth = request.headers['Authorization'].strip("Basic").strip(" ");

    if encoded_auth != "apikey_for_external_platform":
        return "Unauthorized", 401

    data = request.get_json(force=True)
    # convert data into dataframe
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    pred = model.predict(data_df)
    proba = model.predict_proba(data_df)

    pred_list = []
    # create prediction response in DataRobot format
    i = 0
    while i < len(pred):
        dict_pred = {'predictionValues': [{"value": proba[i][0], "label": 0}, {"value": proba[i][1], "label": 1}],
                     'prediction': int(pred[i])}
        pred_list.append(dict_pred)
        i += 1

    return jsonify(data=pred_list)


if __name__ == '__main__':
    app.run()

