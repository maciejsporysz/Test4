# Import required libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load and prepare the model
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from request
    features = request.json['features']
    
    # Make prediction
    prediction = model.predict([features])
    
    return jsonify({'prediction': int(prediction[0]),
                   'class_name': iris.target_names[prediction[0]]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)
