import flask
from flask import jsonify
import pickle
import pandas as pd

def predict(temperature, humidity):
  classes = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
  model = pickle.load(open('ML_model.pkl','rb'))
  model.predict([[temperature, humidity]])
  probability = max(max(model.predict_proba([[temperature, humidity]])))
  return probability

# Initialise the Flask app
app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    return ("Seed The Rise ML Server Running...")

# Set up route for prediction
@app.route('/predict', methods=['POST'])
def main():
    
    if flask.request.method == 'POST':
        temperature = flask.request.form['temperature']
        humidity = flask.request.form['humidity']

        # Get predictions
        prediction = predict(temperature, humidity)
    
        return jsonify(health=prediction)

if __name__ == '__main__':
    app.run()