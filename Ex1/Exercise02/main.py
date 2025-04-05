# -*- coding: utf-8 -*-
"""
#from https://learnopencv.com/camera-calibration-using-opencv/
"""

import cv2
import numpy as np
import os
import glob
import time

# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
outFilePath = "recordings"
numOfFrames = 10  # frames to record
timeDelayInSecs = 0.8

inVideoPath = 0

capture = cv2.VideoCapture(inVideoPath)
if not capture.isOpened:
    print('unable to open video: ' + args.input)
    exit(0)

frameCount = 0

while frameCount < numOfFrames:
    ret, frame = capture.read()
    if frame is None:
        break

    frameCopy = frame.copy()
    gray = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
    filePathToWrite = outFilePath + 'img' + str(frameCount) + ".png"
    cv2.imwrite(filePathToWrite, frameCopy)
    frameCount = frameCount + 1
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    #    """
    #    If desired number of corner are detected,
    #    we refine the pixel coordinates and display
    #    them on the images of checker board
    #    """

    if ret == True:
        print('checkerboard found')
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        frameCopy = cv2.drawChessboardCorners(frameCopy, CHECKERBOARD, corners2, ret)
        filePathToWrite = outFilePath + 'imgCB' + str(frameCount) + ".png"
        cv2.imwrite(filePathToWrite, frameCopy)
        cv2.imshow('img', frameCopy)
    else:
        print('checkerboard NOT found')
    time.sleep(timeDelayInSecs)

cv2.destroyAllWindows()

h, w = frameCopy.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret_RMSerr, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# RMS err should be between 0.1 and 1 pixel
if ret_RMSerr < 2.0:
    print('CAMERA CALIBRATED!!! ERR=' + str(ret_RMSerr))
else:
    print('camera calibration FAILED!! ERR=' + str(ret_RMSerr))

print("Camera matrix : \n")
print(mtx)
print("lens distortion : \n")
print(dist)
print('extrinsic positions for ALL detected shapes')
print("ROTATION rvecs : \n")
print(rvecs)
print("TRANSLATION tvecs : \n")
print(tvecs)