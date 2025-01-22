import cv2
import numpy as np
from ultralytics import YOLO  

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
    
    def detect(self, image):
        results = self.model(image)
        detections = []
        
        for r in results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = r
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(confidence),
                'class_id': int(class_id)
            })
        
        return detections

def detect_objects(image):
    detector = ObjectDetector()
    return detector.detect(image)