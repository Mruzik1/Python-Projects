import cv2 as cv
import numpy as np


new_image = np.zeros((400, 400, 3), dtype = np.uint8)

cv.rectangle(new_image, (100, 50), (200, 300), (241, 43, 100), -1)
cv.circle(new_image, (153, 362), 25, (0, 255, 20), 3)
cv.line(new_image, (200, 300), (153, 362), (0, 0, 255), 5)
cv.putText(new_image, 'Hello World', (50, 100), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.8, (15, 255, 128), 3)

cv.imshow('test', new_image)
cv.waitKey(0)