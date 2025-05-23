import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt 

# ----------------- Sobel Kernels ---------------------
sobelKernelVertical = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

sobelKernelHorizontal = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

sobelKernelDiagonal135 = np.array([
    [0, 1, 2],
    [-1, 0, 1],
    [-2, -1, 0]
])

sobelKernelDiagonal45 = np.array([
    [2, 1, 0],
    [1, 0, -1],
    [0, -1, -2]
])



def rgbImgToGray(inRGBImg):
    gray = cv2.cvtColor(inRGBImg, cv2.COLOR_BGR2GRAY)
    return gray


def sobel(imgGray):
    # remove noise
    img = cv2.GaussianBlur(imgGray, (5, 5), 0)

    # Apply Sobel Masks
    resultH = cv2.filter2D(img, cv2.CV_32F, sobelKernelHorizontal)
    resultV = cv2.filter2D(img, cv2.CV_32F, sobelKernelVertical)
    resultD45 = cv2.filter2D(img, cv2.CV_32F, sobelKernelDiagonal45)
    resultD135 = cv2.filter2D(img, cv2.CV_32F, sobelKernelDiagonal135)
    
    #TODO: combine the images
    img = np.sqrt(pow(resultH, 2) + pow(resultV, 2) + pow(resultD45, 2) + pow(resultD135, 2))

    return img


def normalize(inImg):
    image = inImg.copy()

    max = np.max(image)
    factor = max / 255.0
    normalized = image / factor

    return normalized

# --------------- hough space functions -----------------------
# -------------------------------------------------------------
# old method name "iterateImage"
# TODO: -> cache in file system
def getAccumulationBuffer(inImg, minThresh):
    global accBuffHeight
    global accBuffWidth
    global radiusHelperArr
    accumulationBuffer = np.zeros((accBuffHeight, accBuffWidth), dtype='float32')

    countFG = 0
    for y in range(inImgHeight):
        for x in range(inImgWidth):
            edgeVal = inImg[x][y]
            if edgeVal >= minThresh:
                countFG += 1

                # draw each position as sinusoidal curve in param space
                for degreeIdx in range(accBuffWidth):
                    rad = (degreeIdx * math.pi) / 180.0
                    sinVal = math.sin(rad) # todo: cache
                    cosVal = math.cos(rad)

                    # now calculate radius from angle via polar coordinate formula
                    radiusIdx = int(x * cosVal + y * sinVal + 0.5) # 0.5 to round it
                    if radiusIdx >= 1.0:
                        accumulationBuffer[radiusIdx, degreeIdx] += 1

    print("accumulation buffer calculated with #FG = " + str(countFG / (inImgWidth * inImgHeight) * 100.0) + "%")

    print('accumulation buffer calculated')
    return accumulationBuffer
 
def findMaximaInHoughSpace(accumulationBuffer) :  
  global accBuffHeight
  global accBuffWidth 
  global degreePrecision
  radiusAtMaxVal = -1
  degreeAtMaxVal = -1
  maxVal = -1
 
  for degreeIdx in range(0, accBuffWidth) :
      for radiusIdx in range(0, accBuffHeight) :
          currVal = accumulationBuffer[radiusIdx][degreeIdx]
          if(currVal > maxVal) :
              maxVal = currVal
              radiusAtMaxVal = radiusIdx
              degreeAtMaxVal = degreeIdx
              print('new MAX found with ' + str(maxVal) +
                    ' at radius ' + str(radiusAtMaxVal) +
                    ' at degree ' + str(degreeAtMaxVal))

  return radiusAtMaxVal, degreeAtMaxVal, maxVal
              
  


# --------------- add lines to image --------------------------
# -------------------------------------------------------------


def addLineToImg(imgRGB, maxDegree, maxRadius, accBuffHeight, accBuffWidth, lineColor):
    print('add line to image called with r=' + str(maxRadius) + " degree= " + str(maxDegree))
    x1, y1, x2, y2 = getLineFromHoughSpace(maxDegree, maxRadius, accBuffHeight, accBuffWidth)
    print('resulting line from (' + str(x1) + "," + str(y1) + ") to (" + str(x2) + ", " + str(y2) + ")")
    cv2.line(imgRGB, (x1, y1), (x2, y2), lineColor, thickness=1)  # bgr color model



def getLineFromHoughSpace(maxDegree, maxRadius, accBuffHeight, accBuffWidth):  
    p1_x = maxRadius
    p1_y = -accBuffWidth
    p2_x = maxRadius
    p2_y = accBuffWidth
    rad = ((90.0 - maxDegree) / 180.0) * math.pi
    p1_x_new = round(p1_x * math.cos(rad) - p1_y * math.sin(rad))
    p1_y_new = round(p1_x * math.sin(rad) + p1_y * math.cos(rad))
    p2_x_new = round(p2_x * math.cos(rad) - p2_y * math.sin(rad))
    p2_y_new = round(p2_x * math.sin(rad) + p2_y * math.cos(rad))
    return (p1_x_new, p1_y_new, p2_x_new, p2_y_new)




# ==================== MAIN ==============================================
def main(inImgRGB, lineColor, minThresh):
    # paths for saving images
   
    imgGray = rgbImgToGray(imgRGB)
    plot1 = plt.imshow(imgGray)
    plt.show()
    
    # get Sobel img
    img_sobel = sobel(imgGray)
    img_normalized = normalize(img_sobel)
    plot2 = plt.imshow(img_normalized)  
    plt.show()

    accumulationBuffer = getAccumulationBuffer(img_normalized, minThresh)       
    accumulationBuffer16 = accumulationBuffer.astype(np.uint16)
   
    accumulationBuffer_normalized = normalize(accumulationBuffer)
    plot3 = plt.imshow(accumulationBuffer_normalized)   
    plt.show()
  
    maxRadius, maxDegree, maxVal  = findMaximaInHoughSpace(accumulationBuffer) #TODO define params globally. Thresh relative to max (e.g. 20% of max value in entire accumulation buffer image)

     # add the lines to the image
    addLineToImg(imgRGB, maxDegree, maxRadius, accBuffHeight, accBuffWidth, lineColor)
    plot4 = plt.imshow(imgRGB)
    plt.show()
       
   

def init(inImgHeight, inImgWidth) :
    global accBuffWidth #to access and manipulate the global variable
    global accBuffHeight   
    accBuffWidth = 180 # 0° to 179°, otherwise 0° and 180° would be redundant
    accBuffHeight = int(np.sqrt(inImgHeight * inImgHeight + inImgWidth * inImgWidth) + 0.5 + 1.0) #plus 0.5 to round it, plus 1 to allow for the 0-line with +/-diag above and below
    print('init() finished')
  

if __name__ == "__main__":    
    lineColor = (255, 0, 0)    
    imgInPath = "houghLines2.png"
    imgRGB = cv2.imread(imgInPath)
    inImgWidth = np.shape(imgRGB)[0]
    inImgHeight = np.shape(imgRGB)[1] 
    minThresh = 40
          
    #calculated in init method:
    accBuffWidth = -1
    accBuffHeight = -1    
    init(inImgHeight, inImgWidth)     
    
    main(imgRGB, lineColor, minThresh)

  