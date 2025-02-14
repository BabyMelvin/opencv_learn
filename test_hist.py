import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r

image = cv2.imread("01-gui-features-in-python/image/test.jpg")

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# 直方图均衡化
equalized_image = cv2.equalizeHist(img)


cv2.imshow("Equalized Image", equalized_image)

# 计算 BGR 各通道的直方图
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)


# 颜色直方图均衡化
b, g, r = cv2.split(image)

#对每个通道进行均衡化
b_eq = cv2.equalizeHist(b)
g_eq = cv2.equalizeHist(g)
r_eq = cv2.equalizeHist(r)

# 合并通道
equalized_image = cv2.merge([b_eq, g_eq, r_eq])

plt.plot(hist)
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()