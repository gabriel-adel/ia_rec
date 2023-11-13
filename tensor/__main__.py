import cv2 
import numpy as np
from icecream import ic
import time

import os 


import absl.logging
absl.logging.set_verbosity("FATAL")
from tensor.ia import detect_obj

def limpa():
    os.system('cls' if os.name == 'nt' else 'clear')
limpa()

cap = cv2.VideoCapture(0)
fps = 24
confidence_threshold = 0.3
width, height = 512,512

while True:
    start_time = cv2.getTickCount()
    
    ret, frame = cap.read()
    
    if not ret:
        continue
    frame = cv2.resize(frame, (width, height))
    detections = detect_obj(frame)
    frame_copy = frame.copy()
    for detection, score in zip(detections['detection_boxes'][0].numpy(), detections['detection_scores'][0].numpy()):
        if score > confidence_threshold:
            x,y,w,h = detection 
            x,y,w,h = int(x * 512), int(y*512), int(w*512), int(h*512)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow("detect", frame_copy)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    end_time = cv2.getTickCount()
    enlapsed_time = (start_time - end_time) / cv2.getTickFrequency()
    time.sleep(max(0,1/fps-enlapsed_time))
cap.release()
cv2.destroyAllWindows()

absl.logging.set_verbosity("")