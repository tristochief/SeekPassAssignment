import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from joblib import dump, load
from PIL import Image, ImageOps
import sys
import time
import math

# ---------------------------
# Parameters & Directories
# ---------------------------
image_size = (224, 224)  # MobileNetV2 input size
dataset_dir = './images/train'  # Replace with your dataset path
model_save_path = './model'  # Directory to save the models

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# ---------------------------
# Custom Progress Bar Function
# ---------------------------
def update_progress(progress, total, prefix=''):
    bar_length = 40  # Length of the progress bar
    block = int(round(bar_length * progress / total))
    progress_percent = round(progress / total * 100, 2)
    text = f'\r{prefix} [{"#" * block + "-" * (bar_length - block)}] {progress_percent}%'
    sys.stdout.write(text)
    sys.stdout.flush()

# ---------------------------
# Function to Resize with Padding
# ---------------------------
def resize_with_padding(img, target_size):
    """Resize an image while maintaining aspect ratio and adding padding."""
    old_size = img.size  # (width, height)
    ratio = float(target_size[0]) / max(old_size)  # Scale by longest side
    new_size = tuple([int(x * ratio) for x in old_size])
    
    # Resize image
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Create a new blank image and paste resized image onto it
    new_img = Image.new("RGB", target_size, (0, 0, 0))  # Black padding
    new_img.paste(img, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    
    return new_img

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
def load_data(dataset_dir):
    X, y = [], []
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        img_names = os.listdir(class_dir)
        total_images = len(img_names)

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")  # Ensure RGB format
                img = resize_with_padding(img, image_size)  # Ensure consistent aspect ratio with padding
                img_array = np.array(img, dtype=np.float32)  # Convert to numpy array
                
                img_array = preprocess_input(img_array)  # Normalize to [-1, 1]
                
                X.append(img_array)
                y.append(label)
            except Exception as e:
                print(f"\nError loading {img_path}: {e}")

            update_progress(i + 1, total_images, prefix=f'Loading {class_name} images')

        print()  # New line after each class

    return np.array(X), np.array(y), class_names

# Load the dataset
X, y, class_names = load_data(dataset_dir)
print(f"Loaded {len(X)} images from {len(class_names)} classes.")

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------
# Feature Extraction with MobileNetV2
# ---------------------------
# Initialize MobileNetV2 for feature extraction
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

def extract_features_batch(X_data, batch_size=32):
    total_batches = int(np.ceil(len(X_data) / batch_size))
    features_list = []

    for i in range(total_batches):
        batch_data = X_data[i * batch_size:(i + 1) * batch_size]  # Preprocessed images
        features = base_model.predict(batch_data, batch_size=batch_size, verbose=0)
        features_flat = features.reshape(features.shape[0], -1)
        features_list.append(features_flat)
        update_progress(i + 1, total_batches, prefix='Extracting features')

    print()  # New line after feature extraction
    return np.concatenate(features_list, axis=0)

print("Extracting features for training data...")
X_train_features = extract_features_batch(X_train)
print("Extracting features for validation data...")
X_val_features = extract_features_batch(X_val)

# ---------------------------
# Ensemble of Simple Classifiers with Grid Search
# ---------------------------
rf = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=600, max_depth=25, 
                              max_features='sqrt', criterion='gini')
et = ExtraTreesClassifier(random_state=42, n_jobs=-1, n_estimators=600, max_depth=25, 
                           max_features='sqrt', criterion='entropy')

ensemble_clf = VotingClassifier(estimators=[('rf', rf), ('et', et)], voting='soft')

param_grid = {
    'rf__n_estimators': [500, 600, 700],
    'rf__max_depth': [20, 25, 30],
    'et__n_estimators': [500, 600, 700],
    'et__max_depth': [20, 25, 30]
}

grid_search = GridSearchCV(
    estimator=ensemble_clf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

print("Performing grid search for hyperparameter tuning with ensemble classifier...")
start_time = time.time()
grid_search.fit(X_train_features, y_train)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Grid search completed in {elapsed_time // 60:.0f} minutes and {elapsed_time % 60:.0f} seconds.")

print(f'Best parameters found: {grid_search.best_params_}')
print(f'Best cross-validation score: {grid_search.best_score_:.2f}')

# ---------------------------
# Save the Model
# ---------------------------
dump(grid_search.best_estimator_, os.path.join(model_save_path, 'ensemble_classifier.joblib'))
print(f"Model saved to {os.path.join(model_save_path, 'ensemble_classifier.joblib')}")

# ---------------------------
# Evaluation on Validation Set
# ---------------------------
y_pred = grid_search.best_estimator_.predict(X_val_features)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# ---------------------------
# Load the Saved Model (For future use)
# ---------------------------
loaded_model = load(os.path.join(model_save_path, 'ensemble_classifier.joblib'))
y_pred_loaded = loaded_model.predict(X_val_features)
loaded_accuracy = accuracy_score(y_val, y_pred_loaded)
print(f"Loaded Model Validation Accuracy: {loaded_accuracy * 100:.2f}%")
