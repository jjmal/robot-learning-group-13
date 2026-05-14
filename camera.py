import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Camera opened:", cap.isOpened())

ret, frame = cap.read()
if ret:
    cv2.imwrite("test_frame.jpg", frame)
    print(f"Saved frame: {frame.shape}")
else:
    print("Failed to read frame")

cap.release()