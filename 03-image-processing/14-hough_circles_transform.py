import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('image/opencv_logo.png', 0)
img = cv2.medianBlur(img, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30
                           , minRadius=0, maxRadius=0)
# around 获得最近的偶数值
circles = np.uint16(np.around(circles))
print(circles.shape)

for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)

    # draw the center of the circle
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
cv2.imshow('detected circles', cimg)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
