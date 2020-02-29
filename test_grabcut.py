from grabcut import GrabCut
import numpy as np
import cv2 as cv

filename = 'messi5.jpg'
img = cv.imread(filename)
mask = np.zeros(img.shape[:2], dtype=np.uint8)
rect = (49, 42, 459, 296)

print(img.shape, mask.shape)

g = GrabCut(img, mask, rect)
