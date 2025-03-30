# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:05:18 2020

@author: P21702
"""

# adapted from https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html

import cv2
import numpy as np

inVideoPath = 0
inVideoPath = "./vtest.avi"

capture = cv2.VideoCapture(inVideoPath)
if not capture.isOpened:
    print('unable to open: ' + args.input)
    exit(0)

frameCount = 0
delayInMS = 10 # can be reduced for exercise

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    frameCopy = frame.copy() #deep copy, atomar

    if frameCount < 1 :
        cumulatedFrame = np.zeros(frameCopy.shape)
        cumulatedFrame = cumulatedFrame + frameCopy
        frameCount = frameCount + 1
    else:
        cumulatedFrame = cumulatedFrame + frameCopy
        frameCount = frameCount + 1


    maxVal = np.max(cumulatedFrame)
    avgVal = np.average(cumulatedFrame)
    print("iter " + str(frameCount) + " max= " + str(maxVal) + " avg= " + str(avgVal))

    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(frameCount), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('Frame', frame)

    avgFrame = cumulatedFrame / (frameCount, frameCount, frameCount)
    maxVal = np.max(avgFrame)
    avgVal = np.average(avgFrame)
    print("iter " + str(frameCount) + " max= " + str(maxVal) + " avg= " + str(avgVal))
    avgFrame = avgFrame.astype('uint8')
    cv2.imshow('avg frame', avgFrame)

    diffFrame = avgFrame - frameCopy
    diffVal = np.abs(diffFrame)
    diffFrame = diffFrame.astype('int8')
    blueImage = diffFrame[:, :, 0]
    greenImage = diffFrame[:, :,1]
    redImage = diffFrame[:, :, 2]
    threshold = 30
    segmentedImgIdx = ((blueImage > threshold) | (redImage > threshold) | (greenImage > threshold))

    binaryResImg = frameCopy.copy()
    binaryResImg[segmentedImgIdx] = 255

    cv2.imshow('segmented image', binaryResImg)

    cv2.imshow('diff frame', diffFrame)


    keyboard = cv2.waitKey(delayInMS)
