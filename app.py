import flask
from flask import jsonify, make_response, request
import pickle
import pandas as pd

def predict_health(temperature, humidity, label):
  classes = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
  model = pickle.load(open('ML_model.pkl','rb'))
  model.predict([[temperature, humidity]])
  index = classes.index(label)
  lst = max(model.predict_proba([[temperature, humidity]]))
  health_probability = lst[index]
  return health_probability

# Initialise the Flask app
app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    response = make_response("Seed The Rise ML Server is Running...", 200)
    response.mimetype = "text/plain"
    return response

# Set up route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if flask.request.method == 'POST':
        request_data = request.get_json()
        temperature = request_data['temperature']
        humidity = request_data['humidity']
        cropname = request_data['crop_name']

        # Get predictions
        prediction = predict_health(temperature, humidity, str(cropname))
        response = make_response(str(prediction), 200)
        response.mimetype = "text/plain"
        return response

if __name__ == '__main__':
    app.run()