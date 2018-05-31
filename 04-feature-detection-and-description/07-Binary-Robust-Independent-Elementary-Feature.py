import cv2
import matplotlib.pylab as plt
import numpy as np


def brief_():
    img = cv2.imread('image/cube.jpg', 0)

    # initiate STAR detector
    star = cv2.FastFeatureDetector_create("STAR")

    # initate BRIEF extrator
    brief = cv2.DescriptorDector_create("BRIEF")

    # find the keypointts with STAR
    kp = star.detect(img, None)

    # compute the descriptors with BRIEF
    kp, des = brief.compute(img, kp)


brief_()
