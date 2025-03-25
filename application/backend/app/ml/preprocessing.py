import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def resize_with_padding(img, target_size):
    """
    Resize an image to a target size while maintaining aspect ratio
    and padding with black background.
    """
    old_size = img.size  # (width, height)
    ratio = float(target_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, (0, 0, 0))  # Black padding
    new_img.paste(img, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    return new_img

def preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Preprocess the input image to match the preprocessing from app5.py.
    - Loads image from bytes.
    - Converts to RGB.
    - Resizes while maintaining aspect ratio and adding padding.
    - Converts to a numpy array.
    - Applies MobileNetV2 preprocessing (scales pixel values to [-1, 1]).
    - Adds a batch dimension.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = resize_with_padding(img, target_size)
    
    # Convert to numpy array and preprocess
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
