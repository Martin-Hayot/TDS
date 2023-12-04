import cv2
import numpy as np

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    # Resize the frame
    frame = cv2.resize(frame, (800, 600))

    # Combine Gaussian blur and Canny edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 150)

    # Thresholding using OpenCV
    red_condition = (frame[:, :, 2] >= frame[:, :, 1] + 20) & (frame[:, :, 2] >= frame[:, :, 0] + 20)
    red_composition = np.where(red_condition, 255, 0).astype(np.uint8)
    red_composition_smoothed = cv2.medianBlur(red_composition, 5)

    countours, hierarchy = cv2.findContours(red_composition_smoothed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hand = cv2.drawContours(frame, countours, -1, (0, 255, 0), 3)
    # Display at a reduced rate
    cv2.imshow('frame', frame)
    cv2.imshow('frame2', red_composition_smoothed)
    cv2.imshow('frame3', edges)
    if cv2.waitKey(10) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
