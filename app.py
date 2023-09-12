# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    cough = request.form.get('cough')
    fever = request.form.get('fever')
    body_aches = request.form.get('body_aches')
    headache = request.form.get('headache')
    nausea = request.form.get('nausea')
    vomiting = request.form.get('vomiting')
    
    # Map user input to binary values
    binary_mapping = {'Yes': 1, 'No': 0}
    user_input = np.array([[
        binary_mapping[cough],
        binary_mapping[fever],
        binary_mapping[body_aches],
        binary_mapping[headache],
        binary_mapping[nausea],
        binary_mapping[vomiting]
    ]])
    
    # Make predictions
    prediction = model.predict(user_input)[0]
    
    # Determine the stage based on the prediction
    stages = {0: 'other', 1: 'Acute', 2: 'Moderate', 2: 'Mild'}  # Updated to handle values beyond 0, 1, 2
    stage = stages.get(prediction, 'Moderate')  # Handle unknown predictions
    
    return render_template('index.html', prediction=f'The predicted stage is {stage} Dengue.')

if __name__ == '__main__':
    app.run(debug=True)
