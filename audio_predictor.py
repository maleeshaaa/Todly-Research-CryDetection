import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model from the saved file
model = load_model('best_model.h5')

# Load the pre-fitted scaler for feature normalization
scaler = np.load('scaler.npy', allow_pickle=True).item()

# Load the label encoder to convert model output indices back to human-readable labels
label_encoder = joblib.load('label_encoder.pkl')

def predict_class(file_path):
    # Define a nested function to extract audio features
    def extract_features(file_path, n_mfcc=100):
        # Load the audio file with its native sample rate
        audio, sample_rate = librosa.load(file_path, sr=None)

        # Extract the MFCC (Mel-frequency cepstral coefficients) features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        # Calculate the first-order differences (delta) of MFCC features
        delta_mfccs = librosa.feature.delta(mfccs)

        # Calculate the second-order differences (delta-delta) of MFCC features
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        # Combine the mean values of MFCC, delta, and delta-delta features into a single feature vector
        combined_features = np.hstack((np.mean(mfccs.T, axis=0),
                                       np.mean(delta_mfccs.T, axis=0),
                                       np.mean(delta2_mfccs.T, axis=0)))

        # Return the combined feature vector
        return combined_features

    # Extract features from the provided audio file
    features = extract_features(file_path)

    # Scale the features using the pre-fitted scaler
    features_scaled = scaler.transform([features])

    # Predict the class probabilities using the trained model
    prediction = model.predict(features_scaled)

    # Get the index of the highest probability class
    predicted_class_index = np.argmax(prediction, axis=1)

    # Get the confidence score, which is the probability of the predicted class
    confidence_score = float(np.max(prediction, axis=1))

    # Convert the predicted class index to the corresponding label
    predicted_class_label = label_encoder.inverse_transform(predicted_class_index)

    # Return the predicted class label and the confidence score
    return predicted_class_label[0], confidence_score
