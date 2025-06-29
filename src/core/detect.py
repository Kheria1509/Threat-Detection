from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Model
# model = YOLO(r'M:/mask/detect/train/weights/best.pt')
model = YOLO(r'M:/mask/detect/fire_smoke_train/weights/best.pt')
# model = YOLO(r'M:/mask/yolo-Weights/yolov8n.pt')

# Object classes
classNames = ["fire", "smoke"]
# classNames = ["weapon"]


# Initialize accuracy counters
total_correct = 0
total_predictions = 0

while True:
    success, img = cap.read()
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
            if classNames[cls] == "fire":
                color = (0, 0, 255)  # Red for masked person
                # total_correct += 1  # Increment correct count for masked person
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            # else:
                # color = (255, 255, 0)  # Blue for unmasked person
                # cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

            total_predictions += 1  # Increment total predictions count

            # cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # Calculate accuracy
    accuracy = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0
    print("Accuracy ---> {:.2f}%".format(accuracy))

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
