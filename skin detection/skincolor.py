import numpy as np
import cv2

cap = cv2.VideoCapture(0)
lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

while(1):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinmask = cv2.inRange(frame, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinmask = cv2.erode(skinmask, kernel, iterations=2)
    skinmask = cv2.dilate(skinmask, kernel, iterations = 2)

    skinmask = cv2.GaussianBlur(skinmask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinmask)
    cv2.imshow('skin', np.hstack([frame, skin]))
    cv2.imshow('mask', skin)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
