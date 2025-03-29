import numpy as np
import cv2
import sys

# from: https://github.com/opencv/opencv/blob/master/samples/python/floodfill.py

inFilePath = "color_monkey.jpg"


class App():

    def update(self, dummy=None):
        if self.seed_pt is None:
            cv2.imshow('floodfill', self.img)
            return

        flooded = self.img.copy()
        self.mask[:] = 0
        lo = cv2.getTrackbarPos('lo', 'floodfill')
        hi = cv2.getTrackbarPos('hi', 'floodfill')
        flags = self.connectivity
        flags |= cv2.FLOODFILL_FIXED_RANGE

        cv2.floodFill(flooded, self.mask, self.seed_pt, (255, 255, 255), (lo,) * 3, (hi,) * 3, flags)
        cv2.circle(flooded, self.seed_pt, 2, (0, 0, 255), -1)
        cv2.imshow('floodfill', flooded)

    def onmouse(self, event, x, y, flags, param):
        if flags & cv2.EVENT_FLAG_LBUTTON:
            self.seed_pt = x, y
            self.update()

    def run(self):

        self.img = cv2.imread(inFilePath)

        if self.img is None:
            print('Failed to load image file:', inFilePath)
            sys.exit(1)

        h, w = self.img.shape[:2]
        self.mask = np.zeros((h + 2, w + 2), np.uint8)
        self.seed_pt = None
        self.fixed_range = True
        self.connectivity = 4

        self.update()
        cv2.setMouseCallback('floodfill', self.onmouse)
        cv2.createTrackbar('lo', 'floodfill', 0, 255, self.update)
        cv2.createTrackbar('hi', 'floodfill', 255, 255, self.update)

        while True:
            ch = cv2.waitKey()

        print('Done')


App().run()
cv2.destroyAllWindows()