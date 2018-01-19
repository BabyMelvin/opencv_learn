import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
    image
        cv2.imread()
        注意cv2是BGR模式，而Matplotlib RGB
            flag:
                -1->cv2.IMREAD_COLOR:加载彩色图片。图像透明度被忽略
                 0->cv2.IMREAD_GRAYSCALE:灰度模式加载
                 1->cv2.IMREAD_UNCHANGED:包含alpha通道
        cv2.imshow()
            显示图片，窗口自适应图片大小。
        cv2.imwrite()
"""

# 加载一个图片
img = cv2.imread('test.jpg', 0)
# 路径不会抛出错误，但print img ->None
# print(img)

# 显示图片
cv2.imshow('image', img)
cv2.waitKey(0)  # 键盘信息
cv2.destroyAllWindows()

# 也可以先建立一个窗口，然后再加载图片
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.imshow('image2', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图片
cv2.imwrite('testgray.png', img)

# 完整程序
img0 = cv2.imread('test.jpg', 0)
cv2.imshow('image3', img0)
k = cv2.waitKey(0) & 0xFF  # 64位系统0
if k == 27:  # 等待ESC
    cv2.destroyAllWindows()
elif k == ord('s'):  # 's'key保存和退出
    cv2.imwrite('testgray2.png', img0)
    cv2.destroyAllWindows()

"""
    matplotlib
        显示图片
        zoom images: 缩放图片，变焦
"""
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # hide tick values on x and y axis
plt.show()
