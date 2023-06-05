import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)

CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data['data'])

    nh = pd.DataFrame.from_dict(data=data['data'])
    print(nh)

    df = pd.read_csv('trans.csv')

    X = df.drop(["fraud"], axis=1).to_numpy()
    y = df["fraud"]

    scaler = StandardScaler()
    X_data = X  # scaler.fit_transform(X)
    y_data = y  # .to_numpy()

    model = LinearRegression()
    model.fit(X_data, y_data)

    pred = model.predict([[22, 1044, 6]])
    print(pred)

    return jsonify({'prediction': pred[0]})


if __name__ == '__main__':
    app.run(port=8000, debug=True)
