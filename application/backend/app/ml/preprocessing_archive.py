import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Preprocess the input image to match the exact preprocessing from app2.py.
    - Loads image from bytes.
    - Converts to RGB.
    - Directly resizes to 224x224 using LANCZOS (same as app2.py).
    - Converts to a numpy array.
    - Applies MobileNetV2 preprocessing (scales pixel values to [-1, 1]).
    - Adds a batch dimension.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Ensure the same resampling method is used
    if hasattr(Image, "Resampling"):
        resample_method = Image.Resampling.LANCZOS
    else:
        resample_method = Image.ANTIALIAS  # For older versions of PIL
    
    # Resize directly to (224, 224)
    img = img.resize(target_size, resample_method)

    # Convert to numpy array and preprocess
    img_array = np.array(img)
    img_array = preprocess_input(img_array)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

