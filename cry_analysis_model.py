import numpy as np
import librosa
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam


# Function to extract audio features from a file
def extract_features(file_path, n_mfcc=100):
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


# Function to load data and labels from a dataset directory
def load_data_and_labels(dataset_path='baby_cry_dataset'):
    # List all label directories in the dataset path
    labels = os.listdir(dataset_path)

    # Initialize lists to store all features and corresponding labels
    all_features = []
    all_labels = []

    # Loop through each label directory
    for label in labels:
        # List all sound files within the current label directory
        sound_files = os.listdir(os.path.join(dataset_path, label))

        # Loop through each sound file
        for sound_file in sound_files:
            # Get the full path of the sound file
            file_path = os.path.join(dataset_path, label, sound_file)

            # Extract features from the audio file
            data = extract_features(file_path)

            # Append the features and corresponding label to the lists if features are successfully extracted
            if data is not None:
                all_features.append(data)
                all_labels.append(label)

    # Convert the lists of features and labels to numpy arrays
    return np.array(all_features), np.array(all_labels)


# Load the features and labels from the dataset
features, labels = load_data_and_labels()

# Encode the labels as integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Convert the encoded labels to categorical format (one-hot encoding)
categorical_labels = to_categorical(encoded_labels)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, categorical_labels, test_size=0.2, random_state=42)

# Initialize and fit the scaler on the training data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Scale the testing data using the previously fitted scaler
X_test_scaled = scaler.transform(X_test)

# Build a sequential neural network model
model = Sequential([
    Dense(1024, input_shape=(X_train_scaled.shape[1],)),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.5),

    Dense(512),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.5),

    Dense(256),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.5),

    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model with Adam optimizer and categorical cross-entropy loss
optimizer = Adam(learning_rate=0.0001)  # Lower learning rate
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data, with validation on a 20% split of the training data
history = model.fit(X_train_scaled, y_train, epochs=500, batch_size=32, validation_split=0.2)

# Plot the training and validation loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f'Test accuracy: {test_acc}')

# Predict the classes for the test data
y_pred = model.predict(X_test_scaled)

# Convert the predicted probabilities to class indices
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert the true labels to class indices
y_true_classes = np.argmax(y_test, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Save the trained model to a file
model.save('best_model.h5')

# Save the scaler for future use
np.save('scaler.npy', scaler)

# Save the label encoder for future use
joblib.dump(label_encoder, 'label_encoder.pkl')
