import cv2
import numpy as np
from keras.models import load_model
import keras

# Load the saved model
model_save_path = "video_classifier_model_lstm.h5"
sequence_model = load_model(model_save_path)
print("Model loaded successfully")

# Define constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

# Function to crop the center square of a frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

# Function to resize and preprocess a frame
def preprocess_frame(frame):
    frame = crop_center_square(frame)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
    frame = frame / 255.0  # Normalize pixel values
    return frame

# OpenCV setup for webcam
cap = cv2.VideoCapture('Violence_house.mp4')  # 0 is the default camera index
if not cap.isOpened():
    print("Error: Unable to open webcam")
    exit()

while True:
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        print("Error: Unable to read frame from webcam")
        break
    
    processed_frame = preprocess_frame(frame)  # Preprocess frame
    processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension
    
    # Perform sequence prediction on the frame
    frame_features, frame_mask = prepare_single_video(processed_frame)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
    
    # Display prediction results on the frame
    top_class_index = np.argmax(probabilities)
    top_probability = probabilities[top_class_index]

    # Display the frame with prediction
    font_color = (0, 255, 0)  # Default font color is green
    if top_class_index == 1 and top_probability > 0.515:  # Check if prediction is class 1 and probability is above 60%
        font_color = (0, 0, 255)  # Change font color to red
    if(top_class_index==1):
        cv2.putText(frame, f"Violence_Pred: {top_class_index} ({top_probability:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, font_color, 2)
    else:    
        cv2.putText(frame, f"Prediction: {top_class_index} ({top_probability:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, font_color, 2)
    cv2.imshow("Webcam", frame)
    
    # Check for key press (press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
