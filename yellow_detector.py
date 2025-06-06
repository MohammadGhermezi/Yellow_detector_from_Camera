import cv2 as cv
import numpy as np
from PIL import Image


def get_limmit(color):
    c=np.uint8([[color]])
    
    hsvc=cv.cvtColor(c,cv.COLOR_BGR2HSV)
    
    lower_limit=hsvc[0][0][0]-10,100,100
    upper_limit=hsvc[0][0][0]+10,255,255
    
    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)

    
    return lower_limit,upper_limit


cam=cv.VideoCapture(0)

while True:    
    ret,frame=cam.read()
    
    hsv_img=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    
    lower,upper=get_limmit([0,255,255])
    
    mask=cv.inRange(hsv_img,lower,upper)
    
    mask_=Image.fromarray(mask)
    bbox=mask_.getbbox()
    
    
    ret,theresh=cv.threshold(mask,200,255,cv.THRESH_BINARY_INV)

    countors,heirachy=cv.findContours(theresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    for cnt in countors:
      if cv.contourArea(cnt) >2:
        x1,y1,w,h=cv.boundingRect(cnt)
        cv.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),1)
   
    # if bbox is not None:
    #     x1,y1,x2,y2=bbox
    #     cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)   
     
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
    
cam.release()
cv.destroyAllWindows()