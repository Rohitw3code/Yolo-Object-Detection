from ultralytics import YOLO
import cv2
model = YOLO('yolov8l.pt')
results = model('cars.jpg',show=True)
print(results)
cv2.waitKey(0)