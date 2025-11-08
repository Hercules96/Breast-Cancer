import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("models/breast_cancer_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    arr = np.array(data).reshape(1, -1)
    result = model.predict(arr)[0]
    return jsonify({"prediction": int(result)})

if __name__ == "__main__":
    app.run(debug=True)
