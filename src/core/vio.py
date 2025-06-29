from keras.models import load_model
from collections import deque
import cv2
import numpy as np

def print_results(model_path):
    print("Loading model ...")
    model = load_model(model_path)
    Q = deque(maxlen=128)
    
    # Open the webcam
    vs = cv2.VideoCapture('FIGHT_PRACTICE.mp4')

    while True:
        # Read a frame from the webcam
        grabbed, frame = vs.read()

        # If the frame was not grabbed, break from the loop
        if not grabbed:
            break

        output = frame.copy()

        # Preprocess the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32") / 255

        # Make predictions on the frame
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)

        # Perform prediction averaging over the current history of previous predictions
        results = np.array(Q).mean(axis=0)
        label = 1 if results > 0.55 else 0

        text_color = (0, 255, 0)  # Default: green
        if label:  # Violence prediction
            text_color = (0, 0, 255)  # Red

        text = "Violence: {}".format(label)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        # Show the output image
        cv2.imshow("Live Violence Detection", output)

        # Check for the 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    print("[INFO] Cleaning up...")
    vs.release()
    cv2.destroyAllWindows()

# Example usage
model_path = "./detect/modelnew.h5"
print_results(model_path)
