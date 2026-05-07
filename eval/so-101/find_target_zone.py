import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Clicked: ({x}, {y})")
        if len(points) == 2:
            import numpy as np
            c = points[0]
            edge = points[1]
            r = int(np.sqrt((c[0]-edge[0])**2 + (c[1]-edge[1])**2))
            print(f"TARGET_CENTER_PX = {c}")
            print(f"TARGET_RADIUS_PX = {r}")

cv2.imshow("Click center then edge of target circle", frame)
cv2.setMouseCallback("Click center then edge of target circle", click)
print("Click 1: center of target circle")
print("Click 2: edge of target circle")
cv2.waitKey(0)
cv2.destroyAllWindows()