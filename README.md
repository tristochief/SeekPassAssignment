# Identity Document Classification

This project implements a simple machine learning model for identity document classification using a provided dataset of images. It covers data preprocessing, model training, evaluation, and deployment. The system is composed of three main components:

- **Data Preprocessing & Model Building:**  
  A Python pipeline that loads and preprocesses images, extracts features using MobileNetV2, trains an ensemble classifier (combining Random Forest and Extra Trees) with Grid Search, and evaluates the model.

- **Backend Service:**  
  A Flask API that receives uploaded images, preprocesses them, extracts features, predicts the document class using the trained model, and returns the result.

- **Frontend UI:**  
  A React application that allows users to upload test images and view the predicted class.

- **Containerization:**  
  The backend and frontend are containerized using Docker, and Docker Compose is used to orchestrate the services.

---

## Assignment Overview

### Task 2

- **Objective:**  
  Build a simple ML model for identity document classification using the provided dataset of images. Create a basic UI (preferably in React) that lets users submit test images and receive a predicted class. Containerize the services with Docker Compose for easy deployment.

- **Dataset:**  
  The dataset (`images.zip`) contains 10 classes with 100 examples each:
  - `alb_id` (ID Card of Albania)
  - `aze_passport` (Passport of Azerbaijan)
  - `esp_id` (ID Card of Spain)
  - `est_id` (ID Card of Estonia)
  - `fin_id` (ID Card of Finland)
  - `grc_passport` (Passport of Greece)
  - `lva_passport` (Passport of Latvia)
  - `rus_internalpassport` (Internal passport of Russia)
  - `srb_passport` (Passport of Serbia)
  - `svk_id` (ID Card of Slovakia)

- **Tasks:**
  1. **Data Preprocessing and Model Building:**
     - Load and preprocess the dataset.
     - Use machine learning libraries (e.g., scikit-learn, TensorFlow) to build a classification model.
     - Train the model using the dataset.
     - Evaluate performance using metrics like precision, recall, and F1-score.
     - Emphasize simplicity and functionality over high performance.

  2. **Create a Simple User Interface:**
     - Develop a React UI that allows users to upload an image.
     - On submission, send the image to the backend for classification.

  3. **Backend Service:**
     - Create a Flask backend service to receive and process the uploaded image.
     - Use the trained model to predict the class.
     - Return the predicted class to the UI.

  4. **Dockerize the Services:**
     - Containerize the ML model, backend, and UI using Docker.
     - Use Docker Compose to manage and run the containers together.

---

## Project Structure

Below is an overview of the project tree structure:

```plaintext
.
├── application
│   ├── backend
│   │   └── app
│   │       ├── ml
│   │       │   ├── model.py                # MLModel class and prediction logic
│   │       │   └── preprocessing.py        # Image preprocessing functions
│   │       ├── routes.py                   # Flask routes for API endpoints
│   │       └── static
│   │           └── model
│   │               └── ensemble_classifier.joblib  # Trained model file
│   └── frontend
│       ├── public
│       └── src                           # React application source code
├── images
│   ├── test                              # Test images organized by class
│   └── train                             # Training images organized by class
├── testing
│   └── model                             # Model directory for evaluation script
└── training
    └── model                             # Contains training scripts and saved model
```


Note: this is a simplification of the actual project structure

## Step 1: Clone the Repository

Clone the repository and navigate into the project directory:


git clone [<repository_url>](https://github.com/tristochief/SeekPassAssignment.git)
cd [<repository_directory>](https://github.com/tristochief/SeekPassAssignment.git)


## Step 2: Prepare the Dataset

Extract the `images.zip` file into the `images` directory. Ensure the directory structure is maintained as follows:

images/
├── train          # Contains training images organized by class
└── test           # Contains testing images organized by class

use split.py on the images folder to produce the test and train images in the right directory.

## Step 3: Train and test the Model

The training script performs these tasks:
- Loads and preprocesses images with a custom resizing function (with padding).
- Normalizes images using MobileNetV2's `preprocess_input`.
- Extracts features with a frozen MobileNetV2 model.
- Trains an ensemble classifier (RandomForest + ExtraTrees) using Grid Search.
- Evaluates the model on a validation set.
- Saves the best model as `ensemble_classifier.joblib` in the `training/model` directory.

To train the model locally,

copy the images folder into the training folder

then run,


cd training
docker build -t my-training-model .
docker run --rm -v "$(pwd)/model:/app/model" my-training-model

To test the model locally,

copy the images folder to the testing folder, then run

then run,

cd testing
docker build -t flask-ml-app .
docker run -p 8000:8000 flask-ml-app

then open in your web browser the url specified by docker, usually 
http://127.0.0.1:8000

## Step 4: Build and Start the Services with Docker Compose

From the root directory, execute:

docker-compose up --build


## Step 5: Use the Application

- **Frontend:**  
  Open your browser and navigate to [http://localhost:3000](http://localhost:3000). Use the file input to upload an image and click **Predict** to receive the classification result.

- **Backend API:**  
  Available endpoints:
  - `GET /api/hello` — Returns a welcome message.
  - `POST /api/predict` — Accepts an image file, processes it, and returns the predicted class in JSON format.

## Technical Details

### Data Preprocessing & Model Building
- **Image Preprocessing:**  
  Images are resized with padding to maintain the original aspect ratio, and normalized using MobileNetV2's `preprocess_input`.
  
- **Feature Extraction:**  
  Features are extracted using a frozen MobileNetV2 model and then flattened for classification.

- **Model Training:**  
  An ensemble classifier is created by combining RandomForest and ExtraTrees classifiers. Hyperparameters (e.g., `n_estimators`, `max_depth`) are tuned with Grid Search using 5-fold cross-validation.

- **Evaluation:**  
  The model is evaluated on a validation set using metrics like accuracy, precision, recall, and F1-score. A confusion matrix is generated for detailed analysis.

## Future Improvements

- **Performance Optimization:**  
  Further fine-tuning of hyperparameters and exploration of advanced models or deep learning techniques.
  
- **Enhanced UI:**  
  Improve the user interface for better usability, additional features, and error handling.
  
- **Monitoring & Logging:**  
  Integrate logging and monitoring tools to track model performance and service health.
  
- **Scalability:**  
  Consider cloud deployment options to ensure high availability and scalability.
