import cv2 as cv


image = cv.imread('img/image1.png')
capture = cv.VideoCapture('img/funky.mp4')        # the argument could be also an integer (if camera is going to be used)


# cv.imshow('img', image)
# cv.waitKey(0)


while True:
    frames_grabbed, frame = capture.read()          # returns: frames_grabbed[bool] - if any frame is grabbed; frame[image matring] - a video frame

    if cv.waitKey(20) == ord('z') or not frames_grabbed:        # if a key 'z' was pressed of if the frames_grabbed is false
        break

    cv.imshow('vid', frame)         # display the frame

capture.release()           # closes video file or capturing device
cv.destroyAllWindows()      # closes all vindows