import cvzone
from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

model = YOLO('yolov8l.pt')

while True:
    success,img = cap.read()
    results = model(img,stream=True)

    for r in results:
        boxes = r.boxes
        # for box in boxes:
        #     x1,y1,x2,y2 = box.xyxy[0]
        #     x1, x2, y1, y2 = int(x1),int(x2),int(y1),int(y2)
        #     cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        #     cvzone.cornerRect()
            # print(x1,x2,y1,y2)

    cv2.imshow('object detection', cv2.flip(img, 1))
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


