import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from joblib import load

class MLModel:
    def __init__(self):
        # Load the ensemble classifier saved as a joblib file
        model_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'model', 'ensemble_classifier.joblib')
        self.model = load(model_path)
        # Initialize MobileNetV2 for feature extraction
        self.base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        self.base_model.trainable = False

    def predict(self, image_array):
        """
        Expects image_array of shape (1, height, width, 3) that has been preprocessed with MobileNetV2's preprocess_input.
        Features are extracted using MobileNetV2, flattened, and then passed to the ensemble classifier for prediction.
        """
        # Extract features using MobileNetV2 and flatten them
        features = self.base_model.predict(image_array, verbose=0)
        features = features.reshape(features.shape[0], -1)
        # Predict using the ensemble classifier
        return self.model.predict(features)
