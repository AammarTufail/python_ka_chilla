# Counting Cars over the bridge
```py
import cv2 as cv
import numpy as np
from time import sleep

width_min=80 #Minimum rectangle width
height_min=80 #Minimum height of rectangle

offset=6 #Permissible error between pixels  

pos_line=570 #position of the counting line

delay= 60 #fps of video

detect = []
cars= 0

	
def page_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv.VideoCapture('resources/cars.mp4')
subtraction = cv.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey,(3,3),5)
    img_sub = subtraction.apply(blur)
    dilat = cv.dilate(img_sub,np.ones((5,5)))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilatada = cv.morphologyEx (dilat, cv. MORPH_CLOSE , kernel)
    dilatada = cv.morphologyEx (dilatada, cv. MORPH_CLOSE , kernel)
    contour,h=cv.findContours(dilatada,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
    cv.line(frame1, (25, pos_line), (1200, pos_line), (255,127,0), 3) 
    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv.boundingRect(c)
        validate_contour = (w >= width_min) and (h >= height_min)
        if not validate_contour:
            continue

        cv.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        center = page_center(x, y, w, h)
        detect.append(center)
        cv.circle(frame1, center, 4, (0, 0,255), -1)

        for (x,y) in detect:
            if y<(pos_line+offset) and y>(pos_line-offset):
                cars+=1
                cv.line(frame1, (25, pos_line), (1200, pos_line), (0,127,255), 3)  
                detect.remove((x,y))
                print("car is detected : "+str(cars))        
       
    cv.putText(frame1, "VEHICLE COUNT : "+str(cars), (450, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv.imshow("Video Original" , frame1)
    cv.imshow("Detector",dilatada)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cv.destroyAllWindows()
cap.release()
```
