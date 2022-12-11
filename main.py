import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
cr=pd.read_csv('Crop_recommendation.csv')

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("crop.pkl", "rb"))
@flask_app.route("/")
def Home():

    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The recommendation crop is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True,host='0.0.0.0',port=8080)
