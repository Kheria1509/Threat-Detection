from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# List of models
model_paths = [
    'detect/train/weights/best.pt',
    'detect/weapondetction1_train/weights/best.pt',
    'detect/weapondetction1_train/weights/best.pt',
    'detect/fire_smoke_train/weights/best.pt'

]

# Load models
models = [YOLO(path) for path in model_paths]

# Class names for each model
classNames_list = [
    ["masked", "person", "masked"],  # Update with the correct class names for model 1
    ["weapon"],  # Update with the correct class names for model 2
    ["weapon"],  # Update with the correct class names for model 3
    ["fire","smoke"]   # Update with the correct class names for model 4
]

# Initialize accuracy counters
total_correct = [0] * len(models)
total_predictions = [0] * len(models)

while True:
    success, img = cap.read()

    # Iterate over models
    for i, (model, classNames) in enumerate(zip(models, classNames_list)):
        results = model(img, stream=True)

        # Coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # Class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # Object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                thickness = 2

                # Set the rectangle color based on class label
                if classNames[cls] == "masked":
                    color = (255, 0, 255)  # Pink for masked person
                elif classNames[cls] == "person":
                    color = (255, 0, 0)  # Blue for person
                elif classNames[cls] == "weapon":
                    color = (210, 4, 45)  # Red for weapon
                elif classNames[cls] == "fire":
                    color = (0, 0, 255)  # Red for weapon
                else:
                    color = (0, 0, 0)  # Default to black

                total_predictions[i] += 1  # Increment total predictions count

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        # Calculate accuracy for each model
        accuracy = (total_correct[i] / total_predictions[i]) * 100 if total_predictions[i] > 0 else 0
        print("Model {} Accuracy ---> {:.2f}%".format(i + 1, accuracy))

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()