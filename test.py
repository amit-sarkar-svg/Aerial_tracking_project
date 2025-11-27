import cv2

img = cv2.imread("cal/images/cam_0/0_000.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (7,6))
print("Found:", ret)
