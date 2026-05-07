import cv2
import numpy as np

# put object in front of the camera
# and adjust the sliders until only the object is visible in the mask 
# => get the HSV values for the object to use in the target detection logic 

cap = cv2.VideoCapture(0)

def nothing(x): pass
cv2.namedWindow("HSV Tuner")
cv2.createTrackbar("H_low", "HSV Tuner", 0, 179, nothing)
cv2.createTrackbar("H_high", "HSV Tuner", 179, 179, nothing)
cv2.createTrackbar("S_low", "HSV Tuner", 0, 255, nothing)
cv2.createTrackbar("S_high", "HSV Tuner", 255, 255, nothing)
cv2.createTrackbar("V_low", "HSV Tuner", 0, 255, nothing)
cv2.createTrackbar("V_high", "HSV Tuner", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hl = cv2.getTrackbarPos("H_low", "HSV Tuner")
    hh = cv2.getTrackbarPos("H_high", "HSV Tuner")
    sl = cv2.getTrackbarPos("S_low", "HSV Tuner")
    sh = cv2.getTrackbarPos("S_high", "HSV Tuner")
    vl = cv2.getTrackbarPos("V_low", "HSV Tuner")
    vh = cv2.getTrackbarPos("V_high", "HSV Tuner")

    mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, sh, vh]))
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"OBJECT_HSV_LOWER = np.array([{hl}, {sl}, {vl}])")
        print(f"OBJECT_HSV_UPPER = np.array([{hh}, {sh}, {vh}])")
        break

cap.release()
cv2.destroyAllWindows()