import numpy as np
class SubImageRegion:
    def __init__(self, startX, startY, width, height, origImgArr):
        self.startX = startX
        self.startY = startY
        self.width = width
        self.height = height
        self.subImgArr = np.array([[0] * width for _ in range(height)])

        for x in range(width):
            for y in range(height):
                if 0 <= (y + startY) < len(origImgArr) and 0 <= (x + startX) < len(origImgArr[0]):
                    self.subImgArr[y][x] = origImgArr[y + startY][x + startX]
                else:
                    print(f"Warning: Coordinates ({x + startX}, {y + startY}) are out of bounds for the original image.")