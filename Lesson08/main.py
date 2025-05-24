import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# from https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
inVideoPath = 0

confidence = 0.25  # for "distance"

patternImgPath = "image-removebg-preview.png"
img2 = cv.imread(patternImgPath, cv.IMREAD_GRAYSCALE)  # trajectory between video stream and pattern as img2

# Initiate SIFT detector
sift = cv.SIFT_create()

capture = cv.VideoCapture(inVideoPath)
if not capture.isOpened:
    print('Unable to open: ' + str(inVideoPath))
    exit(0)

while True:
    print('processing frame')
    ret, frame = capture.read()
    if frame is None:
        break

    img1 = frame.copy()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print('features detected')

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]  # init with "0" in mask

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < confidence * n.distance:
            matchesMask[i] = [1, 0]  # good ones get "1" in mask

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,  # use mask for drawing
                       flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv.imshow('res img', img3)
    keyboard = cv.waitKey(2000)
    if keyboard == 'q' or keyboard == 27:
        exit(-1)
