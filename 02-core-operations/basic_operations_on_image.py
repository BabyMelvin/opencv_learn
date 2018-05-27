import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    目标：
        1.获得像素值，并且修改他们
        2.获得图片属性
        3.设置图片区域
        4.分割和合并图像
"""
# 1.获取修改像素值
img = cv2.imread('test.jpg')
px = img[100, 100]
print(px)

# 获取一个蓝像素 BGR
blue = img[100, 100, 0]
print(blue)

# 修改像素值
img[100, 100] = [255, 255, 255]
print(img[100, 100])

# 获得RED值
red = img.item(10, 10, 2)
print(red)
# 修改RED值
img.itemset((10, 10, 2), 100)
img.item(10, 10, 2)

# 获得图片属性 400x300x3
print(img.shape)

# 获取所有的像素点数 360000 占的字节数
print(img.size)

# 数据类型
print(img.dtype)  # dtype=uint8

