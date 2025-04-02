# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:23:19 2020

@author: p21702
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt

inVideoPath = 0

#activate if using video instead of image
inVideoPath = "faceDetectionSampleVid.mp4"
inImgPath = "emotionImages\\angry.png"

eyeCascadePath = "haarcascade_eye_tree_eyeglasses.xml"
frontalFaceCascadePath = "haarcascade_frontalface_alt.xml"


# 1. Load the cascades
face_cascade = cv2.CascadeClassifier(frontalFaceCascadePath)
eye_cascade = cv2.CascadeClassifier(eyeCascadePath)

useInputImg = False #True

delayInMS = 100

if (useInputImg) : #process single image
    img = cv2.imread(inImgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
      img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = img[y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale(roi_gray)
      for (ex,ey,ew,eh) in eyes:
         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img with detections',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
else : #process video
   capture = cv2.VideoCapture(inVideoPath)
   if not capture.isOpened:
      print('Unable to open: ' + str(inVideoPath))
      exit(0)
   while True:
     print('processing frame')
     ret, frame = capture.read()
     if frame is None:
        print('no frame loaded') 
        break

     img1 = frame.copy()
     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
     faces = face_cascade.detectMultiScale(gray1, 1.3, 5)
     for (x,y,w,h) in faces:
       img1 = cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
       roi_gray = gray1[y:y+h, x:x+w]
       roi_color = img1[y:y+h, x:x+w]
       eyes = eye_cascade.detectMultiScale(roi_gray)
       for (ex,ey,ew,eh) in eyes:
         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

     cv2.imshow('img with detections from video',img1)
     keyboard = cv2.waitKey(delayInMS)
     if keyboard == 'q' or keyboard == 27:
        exit(-1)
     
     