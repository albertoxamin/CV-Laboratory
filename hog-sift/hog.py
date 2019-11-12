import cv2
import numpy as np

cap = cv2.VideoCapture("../material/Video.mp4")
hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while cap.isOpened():
    ret, frame = cap.read()
    (rects, weights) = hog.detectMultiScale(frame)
    for rect in rects:
        cv2.rectangle(frame, (rect[0], rect[1]),
                      (rect[0]+rect[2], rect[1]+rect[3]), (255.0, 0.0, 255.0), thickness=3)

    cv2.imshow("hog detected", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
