import cv2 as cv
import numpy
import os
import cvlib as cvlib

def VideoFromWeb():
    
    #cap = cv.VideoCapture('http://192.168.86.66/video') # reading from ESP32
    cap = cv.VideoCapture(0)  # reading from web camera
    
    if not cap.isOpened():
        exit()
    
    while True:
        ret, frame = cap.read()
        bbox, label,conf  = cvlib.detect_common_objects(frame)        
        output_image = cvlib.object_detection.draw_bbox(frame,bbox,label,conf)
        if ret:
            cv.namedWindow("MyWindow",cv.WND_PROP_FULLSCREEN)
            cv.setWindowProperty('MyWindow',cv.WND_PROP_FULLSCREEN,0)
            cv.imshow('frame',frame)
            
        if cv.waitKey(1)==ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()


if __name__=='__main__':
    VideoFromWeb()