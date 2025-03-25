from flask import Blueprint, request, jsonify
from .ml.model import MLModel
from .ml.preprocessing import preprocess_image

# Mapping of prediction indices to labels
LABELS = {
    0: 'alb_id',
    1: 'aze_passport',
    2: 'esp_id',
    3: 'est_id',
    4: 'fin_id',
    5: 'grc_passport',
    6: 'Iva_passport',
    7: 'rus_internalpassport',
    8: 'srb_passport',
    9: 'svk_id'
}

main = Blueprint('main', __name__)
ml_model = MLModel()

@main.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message='Hello from Flask!')

@main.route('/api/predict', methods=['POST'])
def predict():
    # Expecting an image file in the request (multipart/form-data)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Preprocess the image (resize, apply MobileNetV2 preprocessing, etc.)
    image_array = preprocess_image(file.read())
    prediction = ml_model.predict(image_array)

    # Convert prediction (a numpy array) to a list
    prediction_list = prediction.tolist()

    # Assuming the model returns a single value, extract it
    prediction_index = prediction_list[0] if isinstance(prediction_list, list) else prediction_list


    # Map the prediction index to a label
    label = LABELS.get(prediction_index, 'Unknown')
    # Convert prediction (a numpy array) to a list for JSON serialization.
    return jsonify({'prediction': label})
