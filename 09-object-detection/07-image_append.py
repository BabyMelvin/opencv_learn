import cv2
import numpy as np

# 1.加载图像
img1 = cv2.imread('frame.jpg')
img2 = cv2.imread('frame.jpg')

# 2.转换为灰度图
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 3.特征点检测
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 4.匹配特征点
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 5.筛选匹配点
good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 6.计算单应性矩阵
if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
else:
    print("Not enough matches are found - %d/%d" % (len(good_matches), 4))
    H = None
    exit()

# 7.图像变换
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
wrapped_image = cv2.warpPerspective(img1, H, (w1 + w2, h1))

# 8.图像拼接
wrapped_image[0:h2, 0:w2] = img2

# 9.显示结果
cv2.imshow('Wrapped Image', wrapped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()