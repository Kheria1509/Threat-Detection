#!/usr/bin/env python3
"""
Basic Detection Example
Demonstrates how to use the basic fire/smoke detection system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ultralytics import YOLO
import cv2
import math

def basic_detection_example():
    """Example of basic fire/smoke detection"""
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    # Load model
    model_path = "../data/detect/fire_smoke_train/weights/best.pt"
    model = YOLO(model_path)
    
    # Class names
    classNames = ["fire", "smoke"]
    
    print("Starting basic detection...")
    print("Press 'q' to quit")
    
    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                
                # Get class
                cls = int(box.cls[0])
                class_name = classNames[cls]
                
                # Draw bounding box
                color = (0, 0, 255) if class_name == "fire" else (255, 0, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, f"{class_name} {confidence:.2f}", 
                           (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Basic Detection', img)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    basic_detection_example() 