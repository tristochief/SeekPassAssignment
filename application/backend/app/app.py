from flask import Flask, request, jsonify
from flask_cors import CORS
from ml.model import MLModel
from ml.preprocessing import preprocess_image

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

ml_model = MLModel()

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message='Hello from Flask!')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Preprocess the image
    image_array = preprocess_image(file.read())
    prediction = ml_model.predict(image_array)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
