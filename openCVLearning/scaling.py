import cv2 as cv


image = cv.imread('img/image1.png')

# image_resized1 = cv.resize(image, (100, 500), interpolation = cv.INTER_NEAREST)
# image_resized2 = cv.resize(image, (100, 500), interpolation = cv.INTER_AREA)
# image_resized3 = cv.resize(image, (100, 500), interpolation = cv.INTER_LINEAR)


def resize(img, scale = None, width = None, height = None):
    new_sizes = tuple()

    if scale != None:
        new_sizes = (int(img.shape[0]*scale), int(img.shape[1]*scale))

    elif width != None:
        new_sizes = (int(img.shape[0]*width/img.shape[1]), int(width))

    elif height != None:
        new_sizes = (int(height), int(img.shape[1]*height/img.shape[0]))
    
    return cv.resize(img, new_sizes, interpolation = cv.INTER_AREA)


image_resized = resize(image, height=20)
cv.imshow('img', image_resized)

# cv.imshow('img1', image_resized1)
# cv.imshow('img2', image_resized2)
# cv.imshow('img3', image_resized3)

cv.waitKey(0)