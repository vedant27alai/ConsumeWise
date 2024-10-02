from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extracting age and chemicals from the request data
    age = int(data['age'])  # Convert age to integer
    chemicals = data['chemicals']  # The chemical input as string
    
    # Process chemicals into a feature array
    # For this example, we assume that the model only takes 'age'
    # If your model was trained with more features, include them as needed.
    # Adjust the features based on your training model
    # Let's assume for now it just uses age as input
    input_data = np.array([[age]])  # Only pass age as input

    # Make prediction
    prediction = model.predict(input_data)

    # Return the prediction result
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
