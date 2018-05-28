import cv2
import numpy as np
import matplotlib.pylab as plt

"""
目标：
    代码不仅正确，还要保证速度
        测试代码的性能
        提高代码性能的建议
        cv2.getTickCount,cv2.getTickFrequency
    python:
        提供time和profile模块
"""


def measuring_performance_with_opencv():
    """
    cv2.getTickCount:获得函数执行的时钟
    cv2.getTickFrequency：获得时钟周期频率，每秒多少时钟周期。
    :return:
    """
    e1 = cv2.getTickCount()
    # your code execution
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print("e1:{0},e2:{1},time:{2}".format(e1, e2, time))


def default_optimization_in_opencv():
    """
    很多OpenCv函数优化使用SSE2,AVX等。也包含未优化代码，可以是用cv2.useOptimized()
    检查是否打开。cv2.setUseOptimized()设置是否优化
    :return:
    """
    # check if optimization is enabled
    print(cv2.useOptimized())
    cv2.setUseOptimized(False)
    print(cv2.useOptimized())


def measuring_performance_in_ipython():
    """
    有时需要比较两个相似操作的性能。ipython给了magic 命令`%timeit`
    :return:
    """
    x = 5


# measuring_performance_with_opencv()
default_optimization_in_opencv()
