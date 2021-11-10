import flask
from flask import jsonify, make_response, request
import pickle
import pandas as pd

def predict_health(temperature, humidity):
  classes = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
  model = pickle.load(open('ML_model.pkl','rb'))
  model.predict([[temperature, humidity]])
  probability = max(max(model.predict_proba([[temperature, humidity]])))
  return probability

# Initialise the Flask app
app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    response = make_response("Seed The Rise ML Server Running...", 200)
    response.mimetype = "text/plain"
    return response

# Set up route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if flask.request.method == 'POST':
        request_data = request.get_json()
        temperature = request_data['temperature']
        humidity = request_data['humidity']

        # Get predictions
        prediction = predict_health(temperature, humidity)
        response = make_response(prediction, 200)
        response.mimetype = "text/plain"
        return response
        # return jsonify(health=prediction)

if __name__ == '__main__':
    app.run()