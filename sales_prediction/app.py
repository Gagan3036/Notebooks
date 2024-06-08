from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('salary_prediction_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input from the form
        experience_years = request.form['experience_years']

        # Check if the input can be converted to a float
        try:
            experience_years = float(experience_years)
        except ValueError:
            return jsonify({'error': 'Please enter a valid number for years of experience'})

        # Check if the input is a float or an integer
        if isinstance(experience_years, float):
            # Make prediction for float input
            prediction = model.predict(np.array([[experience_years]]))[0]
            return jsonify({'prediction': prediction})
        elif isinstance(experience_years, int):
            # Make prediction for integer input
            prediction = model.predict(np.array([[experience_years]]))[0]
            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'Please enter a valid number for years of experience'})

    except Exception as e:
        # Return error message if prediction fails
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
