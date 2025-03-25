from flask import Flask, render_template_string, send_from_directory
import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

app = Flask(__name__)

# Initialize MobileNetV2 for feature extraction (as used in training)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

def resize_with_padding(img, target_size):
    old_size = img.size  # (width, height)
    ratio = float(target_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, (0, 0, 0))  # Black padding
    new_img.paste(img, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    return new_img

def get_class_names(test_dir):
    return sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

def load_test_images_and_labels(test_dir):
    images, filenames, true_labels = [], [], []
    class_names = get_class_names(test_dir)
    
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, fname)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = resize_with_padding(img, (224, 224))
                    img_array = np.array(img, dtype=np.float32)
                    img_array = preprocess_input(img_array)
                    images.append(img_array)
                    filenames.append(os.path.join(class_name, fname))
                    true_labels.append(class_names.index(class_name))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    return np.array(images), filenames, np.array(true_labels), class_names

def extract_features_from_images(images):
    features = base_model.predict(images, verbose=0)
    return features.reshape(features.shape[0], -1)

# Route to serve test images (for thumbnails)
@app.route('/test_images/<path:filename>')
def test_images(filename):
    project_root = os.path.abspath(os.path.dirname(__file__))
    test_dir = os.path.join(project_root, 'images', 'test')
    return send_from_directory(test_dir, filename)

@app.route('/')
def index():
    # Build absolute path for model file
    project_root = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(project_root, 'model', 'ensemble_classifier.joblib')
    if not os.path.exists(model_path):
        return f"Model file not found at: {model_path}"
    ensemble_model = load(model_path)
    
    # Get test directory path
    test_dir = os.path.join(project_root, 'images', 'test')
    
    # Load test images, filenames, true labels, and class names
    X_test, filenames, true_labels, class_names = load_test_images_and_labels(test_dir)
    if X_test.size == 0:
        return "No test images found."
    
    # Extract features from test images (same as training)
    X_test_features = extract_features_from_images(X_test)
    
    # Get predictions from the ensemble classifier
    predicted = ensemble_model.predict(X_test_features)
    if predicted.ndim > 1:
        predicted_classes = predicted.argmax(axis=1)
    else:
        predicted_classes = predicted.tolist()
    
    # Compute accuracy, classification report, and confusion matrix
    accuracy = accuracy_score(true_labels, predicted_classes)
    report_dict = classification_report(true_labels, predicted_classes, target_names=class_names, output_dict=True)
    cm = confusion_matrix(true_labels, predicted_classes)
    
    # Build HTML for classification report table
    report_html = "<h2>Classification Report</h2>"
    report_html += (
        "<table border='1' cellpadding='5'>"
        "<tr>"
        "<th>Label Index</th>"
        "<th>Label Name</th>"
        "<th>Precision</th>"
        "<th>Recall</th>"
        "<th>F1-Score</th>"
        "<th>Support</th>"
        "<th>% Correct</th>"
        "</tr>"
    )
    for idx, label in enumerate(class_names):
        if label in report_dict:
            metrics = report_dict[label]
            percent_correct = metrics.get('recall', 0) * 100
            report_html += (
                f"<tr><td>{idx}</td>"
                f"<td>{label}</td>"
                f"<td>{metrics.get('precision', 0):.2f}</td>"
                f"<td>{metrics.get('recall', 0):.2f}</td>"
                f"<td>{metrics.get('f1-score', 0):.2f}</td>"
                f"<td>{metrics.get('support', 0)}</td>"
                f"<td>{percent_correct:.2f}%</td></tr>"
            )
    report_html += f"<tr><td colspan='7'><strong>Overall Accuracy: {accuracy:.2f}</strong></td></tr>"
    report_html += "</table>"
    
    # Build HTML for confusion matrix table (showing percentage per true label)
    cm_html = "<h2>Confusion Matrix (%)</h2>"
    cm_html += "<table border='1' cellpadding='5'>"
    cm_html += "<tr><th>True \\ Predicted</th>"
    for label in class_names:
        cm_html += f"<th>{label}</th>"
    cm_html += "</tr>"
    for i, row in enumerate(cm):
        row_sum = row.sum()
        cm_html += f"<tr><th>{class_names[i]}</th>"
        for val in row:
            perc = (val / row_sum * 100) if row_sum != 0 else 0
            cm_html += f"<td>{perc:.2f}%</td>"
        cm_html += "</tr>"
    cm_html += "</table>"
    
    # Collect case examples for each class (correct and misclassified)
    correct_examples = {i: [] for i in range(len(class_names))}
    wrong_examples = {i: [] for i in range(len(class_names))}
    for idx, (true, pred) in enumerate(zip(true_labels, predicted_classes)):
        # Get filename from our manually loaded list
        filename = filenames[idx]
        if true == pred:
            correct_examples[true].append(filename)
        else:
            wrong_examples[true].append((filename, pred))
    
    # Build HTML for case examples with image thumbnails
    examples_html = "<h2>Case Examples for Each Class</h2>"
    for i in range(len(class_names)):
        examples_html += f"<h3>{i}: {class_names[i]}</h3>"
        examples_html += "<strong>Correctly Classified:</strong><ul>"
        if correct_examples[i]:
            for example in correct_examples[i]:
                img_url = f"/test_images/{example}"
                examples_html += (
                    f"<li>{example}<br>"
                    f"<img src='{img_url}' style='max-width:150px; height:auto;'></li>"
                )
        else:
            examples_html += "<li>No correct examples.</li>"
        examples_html += "</ul>"
        examples_html += "<strong>Misclassified:</strong><ul>"
        if wrong_examples[i]:
            for example, pred in wrong_examples[i]:
                img_url = f"/test_images/{example}"
                examples_html += (
                    f"<li>{example} (predicted as {class_names[pred]})<br>"
                    f"<img src='{img_url}' style='max-width:150px; height:auto;'></li>"
                )
        else:
            examples_html += "<li>No misclassified examples.</li>"
        examples_html += "</ul>"
    
    # Combine all parts into the final HTML page
    html = f"""
    <html>
    <head><title>Model Evaluation</title></head>
    <body>
      {report_html}
      {cm_html}
      {examples_html}
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
