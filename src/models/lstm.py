


import cv2
import numpy as np
from keras.models import load_model
from keras.applications import InceptionV3
from keras.preprocessing import image as keras_image
import keras

# Load the saved model
model_save_path = "video_classifier_model_lstm.h5"
sequence_model = load_model(model_save_path)
print("Model loaded successfully")

#Define hyperparameters

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100

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
            # Preprocess and resize frame to (224, 224)
            processed_frame = preprocess_frame(batch[j])
            # Expand dimensions to add channel dimension
            processed_frame = np.expand_dims(processed_frame, axis=0)
            # Predict features using the feature_extractor
            frame_features[i, j, :] = feature_extractor.predict(processed_frame)
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask
# Function to preprocess frames for model prediction
def preprocess_frame(frame):
    # Resize frame to match model input size
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0  # Normalize pixel values
    return frame



# Function to perform sequence prediction on frames
def sequence_prediction_on_frames(frames):
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
    return probabilities

# OpenCV setup for webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera index
if not cap.isOpened():
    print("Error: Unable to open webcam")
    exit()

while True:
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        print("Error: Unable to read frame from webcam")
        break
    
    processed_frame = preprocess_frame(frame)  # Preprocess frame
    probabilities = sequence_prediction_on_frames(processed_frame)  # Perform sequence prediction
    
    # Display prediction results on the frame
    class_vocab = label_processor.get_vocabulary()
    top_class_index = np.argmax(probabilities)
    top_class_label = class_vocab[top_class_index]
    top_probability = probabilities[top_class_index]

    cv2.putText(frame, f"Prediction: {top_class_label} ({top_probability:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Webcam", frame)
    
    # Check for key press (press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()










































# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.preprocessing import image

# # Load the model architecture and weights
# model_path = r'M:\mask\detect\lstm'
# sequence_model = load_model(os.path.join(model_path, 'video_classifier'))

# # Define constants
# MAX_SEQ_LENGTH = 20  # Assuming a maximum sequence length of 20
# NUM_FEATURES = 2048  # Assuming the ResNet50 model is used as a feature extractor

# # Define the feature extractor
# feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# def load_video_frames(video_path):
#     # Load and preprocess frames from the video
#     frames = []
#     # Your code to load frames from the video
#     return frames

# def prepare_single_video(frames):
#     frames = frames[None, ...]
#     frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
#     frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

#     for i, batch in enumerate(frames):
#         video_length = batch.shape[0]
#         length = min(MAX_SEQ_LENGTH, video_length)
#         for j in range(length):
#             img = image.img_to_array(frames[j])
#             img = preprocess_input(img)
#             frame_features[i, j, :] = feature_extractor.predict(img[None, ...])
#         frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

#     return frame_features, frame_mask

# def sequence_prediction(frames):
#     class_vocab = label_processor.get_vocabulary()
#     frame_features, frame_mask = prepare_single_video(frames)
#     probabilities = sequence_model.predict([frame_features, frame_mask])[0]

#     for i in np.argsort(probabilities)[::-1]:
#         print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")

# # Load frames from the test video
# test_video_path = 'Violence_house.mp4'
# test_frames = load_video_frames(test_video_path)

# # Perform sequence prediction on the test frames
# sequence_prediction(test_frames)
