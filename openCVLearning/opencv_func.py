import cv2 as cv
import numpy as np


# img = np.random.randint(0, 256, (400, 600, 3), dtype = np.uint8)
img = cv.imread('img/anime_face.png')

# Converting to grayscale
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Bluring the image
blur_img = cv.GaussianBlur(img, (3, 99), cv.BORDER_DEFAULT)

# Edge Cascade
edge_img = cv.Canny(img, 100, 150)

# Dilating the image (uses the edged image)
dilated_img = cv.dilate(edge_img, (3, 3), iterations = 2)

# Eroding the image (just reverses the dilating, not perfect tho)
eroded_img = cv.erode(dilated_img, (3, 3), iterations = 2)

# Cropping the image
cropped_img = img[20:200, 100:300]

cv.imshow('test', cropped_img)
cv.waitKey(0)