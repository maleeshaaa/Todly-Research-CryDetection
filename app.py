from audio_predictor import predict_class
from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
from dateutil import relativedelta

# Initialize Flask app
app = Flask(__name__)

# Constants for audio processing and model
SAMPLE_RATE = 22050  # Sampling rate for audio files
N_MFCC = 100  # Number of MFCC features to extract
MODEL_PATH = '../cry-analysis/best_model.h5'  # Path to the trained model

# Load the trained model from the specified path
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize the feature scaler
scaler = MinMaxScaler()


# Load audio files from the dataset and extract features to fit the scaler.
# This function reads audio files, extracts MFCC features, and returns them.
def load_data_for_scaler(dataset_path):
    labels = os.listdir(dataset_path)  # List of label directories
    features = []
    for label in labels:
        sound_files = os.listdir(os.path.join(dataset_path, label))  # List of audio files for each label
        for sound_file in sound_files:
            file_path = os.path.join(dataset_path, label, sound_file)  # Full path to the audio file
            audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)  # Load audio file
            mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)  # Extract MFCC features
            features.append(np.mean(mfccs.T, axis=0))  # Average MFCCs and append to features list
    return np.array(features)


# Initialize scaler with the dataset
dataset_path = '../cry-analysis/baby_cry_dataset'
features = load_data_for_scaler(dataset_path)
scaler.fit(features)  # Fit the scaler with the extracted features


# Extract MFCC features from the provided audio data.
def extract_features(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return np.mean(mfccs.T, axis=0)  # Average MFCCs


# Preprocess the audio file: load audio, extract features, and scale them.
def preprocess_audio(file):
    audio, _ = librosa.load(file, sr=SAMPLE_RATE)  # Load audio file
    features = extract_features(audio)  # Extract features
    return scaler.transform([features])  # Scale features using the pre-fitted scaler


# In-memory storage for predictions
predictions = []


# Capitalize and format the predicted class
def format_class_name(class_name):
    # Replace underscores with spaces
    formatted = class_name.replace('_', ' ')
    # Capitalize the first letter of each word
    return formatted.title()


# Predict the class of the uploaded audio file.
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400  # Error if no file part in the request

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400  # Error if no file selected

    if file:
        file_path = 'buffer.wav'  # Specify the path to save the file as 'buffer.wav'
        file.save(file_path)  # Save the uploaded file, overwriting any existing 'buffer.wav'

        # Predict the class and confidence score of the uploaded file
        class_predict, confidence_score = predict_class(file_path)  # Call predict_class to get class and confidence
        class_predict_formatted = format_class_name(class_predict)  # Format the class name

        # Print the prediction and confidence score for debugging (optional)
        print(f"Predicted class: {class_predict_formatted}, Confidence score: {confidence_score:.2f}")

        # Store the prediction with the current timestamp
        predictions.append({
            'timestamp': datetime.now().isoformat(),
            'predicted_class': class_predict_formatted,
            'confidence_score': confidence_score
        })

        # Return the predicted class and confidence score as a JSON response
        return jsonify({
            'predicted_class': class_predict_formatted,
            'confidence_score': confidence_score
        }), 200


# New route to get the prediction for 'buffer.wav'
@app.route('/get-prediction', methods=['GET'])
def get_prediction():
    file_path = 'buffer.wav'
    if not os.path.exists(file_path):
        return jsonify({'error': 'No audio file found'}), 404  # Error if file not found

    # Predict the class and confidence score of the 'buffer.wav' file
    class_predict, confidence_score = predict_class(file_path)  # Call predict_class to get class and confidence
    class_predict_formatted = format_class_name(class_predict)  # Format the class name

    # Return the predicted class and confidence score as a JSON response
    return jsonify({
        'predicted_class': class_predict_formatted,
    }), 200


# Convert timestamp to human-readable relative time
def relative_time(timestamp):
    now = datetime.now()
    timestamp = datetime.fromisoformat(timestamp)
    delta = relativedelta.relativedelta(now, timestamp)

    if delta.years > 0:
        return f"{delta.years} year{'s' if delta.years > 1 else ''} ago"
    elif delta.months > 0:
        return f"{delta.months} month{'s' if delta.months > 1 else ''} ago"
    elif delta.days > 0:
        if delta.days == 1:
            return "Yesterday"
        else:
            return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    elif delta.hours > 0:
        return f"{delta.hours} hour{'s' if delta.hours > 1 else ''} ago"
    elif delta.minutes > 0:
        return f"{delta.minutes} minute{'s' if delta.minutes > 1 else ''} ago"
    else:
        return "Just Now"


# GET all predictions with timestamps
@app.route('/predictions', methods=['GET'])
def get_predictions():
    # Convert timestamps to relative time and format class names
    formatted_predictions = [
        {
            'timestamp': relative_time(pred['timestamp']),  # Convert timestamp to relative time
            'predicted_class': format_class_name(pred['predicted_class']),  # Format class name
            'confidence_score': pred['confidence_score']  # Include the confidence score
        }
        for pred in predictions
    ]
    return jsonify(formatted_predictions), 200


# Simple index route to verify that the server is running.
@app.route('/')
def index():
    return "Welcome to the Cry Analysis Model API!", 200


# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
