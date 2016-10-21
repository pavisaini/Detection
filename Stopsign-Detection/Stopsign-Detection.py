import numpy as np
import cv2

stopsign_cascade = cv2.CascadeClassifier('stop_sign.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    font = cv2.FONT_HERSHEY_SIMPLEX
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stopsign = stopsign_cascade.detectMultiScale(gray, 1.2, 5)
        
    for (x,y,w,h) in stopsign:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.putText(img,'Stop Sign',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)




    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()